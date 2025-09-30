# -*- coding: utf-8 -*-
"""
FunASR 中文 ASR + 跨录音说话人指派（只用 FunASR，自带 spk/sv 模型）

流程：
  1) ASR + 分句 + 本地说话人（FunASR sentence_info.spk -> 规范化为 SPEAKER_XX）
  2) 按“本地说话人”聚合其时间段，最小窗口 >= SPK_EMB_MIN_SECONDS（默认1.5s），提取 SV 嵌入
  3) 与“参考库”(gallery)做余弦相似度匹配；高于阈值 -> 指派；否则创建新全局说话人并入库
  4) 返回结构包含：
     - segments: [{"text","start_time","end_time","speaker","speaker_name",...}, ...] （speaker 为全局ID）
     - speaker_texts: {"global_spk_1":"...", ...}
     - name_texts: {"Alice":"...", "Bob":"..."}
     - dialogue_lines: ["Alice: 你好……", "Bob: ……"]
     - speaker_mapping: { "local_to_global": {...}, "global_id_to_name": {...}, ... }

可选环境变量：
  FUNASR_ASR_MODEL, FUNASR_VAD_MODEL, FUNASR_PUNC_MODEL, FUNASR_SPK_MODEL
  FUNASR_SV_MODEL
  FUNASR_DEVICE=cuda|cpu|auto        (默认 auto)
  FUNASR_SV_DEVICE=cpu|cuda|auto     (默认继承 FUNASR_DEVICE；建议设为 cpu)
  FUNASR_BATCH_SIZE_S=300
  FUNASR_HUB=hf|ms|auto              (默认 hf)
  SPK_MATCH_THRESHOLD=0.3           （相似度阈值；命中>=阈值）
  SPK_GALLERY_PATH=./speaker_gallery.json
  SPK_EMB_MIN_SECONDS=1.5            （同一说话人聚合后每个窗口的最短秒数）
  SPK_EMB_MAX_GAP=0.3                （同一说话人的相邻片段若间隔<=该秒数则拼接）
  SPK_DEBUG=0/1                      （1则打印每个本地说话人的Top-3相似度）

使用要点：
  - 先 add_reference(...) 登记参考（可传 name），再 transcribe_audio(...)。
  - 如需“仅用这几个参考匹配”，transcribe_audio(..., restrict_gallery_ids=[gid1, gid2])。
"""

import os
import re
import sys
import json
import uuid
import logging
import tempfile
import threading
import subprocess
from collections import defaultdict

try:
    import numpy as np
except Exception:
    raise ImportError("需要 numpy: pip install numpy")

try:
    import jieba
    JIEBA_OK = True
except Exception:
    JIEBA_OK = False

from funasr import AutoModel
import torch

# -------- 日志强制配置（避免被外部运行器覆盖） --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("funasr_service")
logger.propagate = False


# ----------------- 工具函数 -----------------

def _to_numpy_f32(x):
    """torch.Tensor(可在GPU) / list / tuple -> numpy.float32 1D/2D"""
    if isinstance(x, torch.Tensor):
        x = x.detach().to('cpu')
        if x.dtype.is_floating_point:
            x = x.float()
        return x.numpy()
    return np.asarray(x, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (na * nb))


def _ensure_wav_16k_mono(in_path: str) -> str:
    """确保音频为16k单声道wav；若不是则用ffmpeg转换到临时文件并返回其路径"""
    tmpdir = tempfile.mkdtemp(prefix="funasr_sv_")
    out_path = os.path.join(tmpdir, "audio_16kmono.wav")
    cmd = [
        "ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000",
        "-vn", "-f", "wav", out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _slice_to_wav(in_path: str, start_s: float, end_s: float) -> str:
    """用 ffmpeg 按 [start, end) 切片成16k单声道wav，返回临时路径"""
    assert end_s > start_s, "end_s must > start_s"
    tmpdir = tempfile.mkdtemp(prefix="funasr_sv_")
    out_path = os.path.join(tmpdir, "slice_16kmono.wav")
    dur = max(0.01, end_s - start_s)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-t", f"{dur:.3f}",
        "-i", in_path, "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


# ----------------- API 响应封装 -----------------

class ApiResponse:
    def __init__(self, headers=None, json_data=None):
        self.headers = headers if headers is not None else {}
        self.json_data = json_data if json_data is not None else {}

    def json(self):
        return self.json_data


# ----------------- 主服务 -----------------

class FunASRService:
    def __init__(self, model_path='./asr_models/'):
        self.model_lock = threading.Lock()
        self.model_path = model_path

        # ===== 模型与参数 =====
        self.asr_model_id = os.getenv(
            "FUNASR_ASR_MODEL",
            os.path.join(self.model_path, "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        )
        self.vad_model_id = os.getenv(
            "FUNASR_VAD_MODEL",
            os.path.join(self.model_path, "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        )
        self.punc_model_id = os.getenv(
            "FUNASR_PUNC_MODEL",
            os.path.join(self.model_path, "iic/punc_ct-transformer_cn-en-common-vocab471067-large"),
        )
        self.spk_model_id = os.getenv(
            "FUNASR_SPK_MODEL",
            os.path.join(self.model_path, "iic/speech_campplus_sv_zh-cn_16k-common"),
        )
        self.sv_model_id = os.getenv("FUNASR_SV_MODEL", self.spk_model_id)

        self.device = os.getenv("FUNASR_DEVICE", "auto")
        self.sv_device = os.getenv("FUNASR_SV_DEVICE", "cpu")  # 建议 "cpu"
        try:
            self.batch_size_s = int(os.getenv("FUNASR_BATCH_SIZE_S", "300"))
        except Exception:
            self.batch_size_s = 300

        self.hub = os.getenv("FUNASR_HUB", "hf")
        self.disable_update = True
        self.disable_pbar = True

        # 匹配阈值 & 参考库持久化路径
        try:
            self.match_threshold = float(os.getenv("SPK_MATCH_THRESHOLD", "0.3"))
        except Exception:
            self.match_threshold = 0.85
        self.gallery_path = os.getenv("SPK_GALLERY_PATH", "./speaker_gallery.json")

        # 嵌入聚合参数
        try:
            self.emb_min_seconds = float(os.getenv("SPK_EMB_MIN_SECONDS", "1.5"))
        except Exception:
            self.emb_min_seconds = 1.5
        try:
            self.emb_max_gap = float(os.getenv("SPK_EMB_MAX_GAP", "0.3"))
        except Exception:
            self.emb_max_gap = 0.3

        # 调试开关
        self.debug = str(os.getenv("SPK_DEBUG", "0")).strip() == "1"

        # ===== 模型句柄 =====
        self.funasr_asr = None        # ASR(+VAD+PUNC+SPK)
        self.funasr_sv = None         # 说话人验证/嵌入

        # ===== 参考说话人库 =====
        # 结构：{"global_spk_1": {"name": "可选", "embedding": [float,...], "count": k}, ...}
        self.gallery = {}
        self._load_gallery()

        logger.info(
            f"[Init] device={self.device}, sv_device={self.sv_device}, hub={self.hub}, "
            f"threshold={self.match_threshold}, gallery={len(self.gallery)}, "
            f"emb_min_seconds={self.emb_min_seconds}, emb_max_gap={self.emb_max_gap}, debug={self.debug}"
        )

    # ---------- 公共 API ----------

    def submit_task(self, file_url, format="wav"):
        task_id = str(uuid.uuid4())
        x_tt_logid = str(uuid.uuid4())
        logger.info(f"[Submit] task_id={task_id}, file={file_url}")
        return task_id, x_tt_logid

    def query_task(self, task_id, x_tt_logid, file_url, format="wav", **kwargs):
        """
        支持可选参数：restrict_gallery_ids=[...] 仅在这些ID中匹配
        """
        headers = {"X-Api-Status-Code": "20000000", "X-Api-Message": "Success", "X-Tt-Logid": x_tt_logid}
        try:
            result = self._transcribe_with_global_assign(file_url, **kwargs)
            return ApiResponse(headers=headers, json_data={"result": result})
        except Exception as e:
            logger.exception("[QueryTask] error")
            headers["X-Api-Status-Code"] = "50000000"
            headers["X-Api-Message"] = str(e)
            return ApiResponse(headers=headers, json_data={"error": str(e)})

    def transcribe_audio(self, file_path, format="wav", **kwargs):
        """
        kwargs 可传：
          - hotword_hint: str
          - restrict_gallery_ids: list[str] 仅在这些全局ID候选中做匹配
        """
        with self.model_lock:
            task_id = str(uuid.uuid4())
            x_tt_logid = str(uuid.uuid4())
            resp = self.query_task(task_id, x_tt_logid, file_path, format, **kwargs)
            if resp.headers.get("X-Api-Status-Code") == "20000000":
                return resp.json()["result"]
            else:
                raise RuntimeError(resp.headers.get("X-Api-Message", "Unknown error"))

    # ---------- 参考库（外部可调用） ----------

    def add_reference(self, audio_path: str, segments=None, label: str = None, name: str = None):
        """
        把一段参考音频登记到“参考库”：
        - audio_path: 参考录音路径
        - segments: 可选的 [(start_s, end_s), ...]；不传就用整段
        - label: 指定写入的全局ID；不传则自动分配新的 global_spk_N
        - name: 人名/备注（可选）
        """
        self._ensure_sv_model()

        abs_path = os.path.abspath(audio_path)
        base_wav = _ensure_wav_16k_mono(abs_path)

        if not segments:
            segments = [(0.0, self._probe_duration(base_wav))]

        embs = []
        for (st, ed) in segments:
            if ed <= st:
                continue
            slice_wav = _slice_to_wav(base_wav, st, ed)
            emb = self._extract_sv_embedding(slice_wav)  # (D,)
            embs.append(emb)

        if not embs:
            raise ValueError("参考音频没有有效语音片段")

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)

        if not label:
            label = self._new_global_id()

        entry = self.gallery.get(label, {"name": name or "", "embedding": None, "count": 0})
        if entry["embedding"] is None:
            entry["embedding"] = mean_emb.tolist()
            entry["count"] = len(embs)
        else:
            old = np.array(entry["embedding"], dtype=np.float32)
            c = entry["count"]
            new = (old * c + mean_emb * len(embs)) / (c + len(embs))
            entry["embedding"] = new.tolist()
            entry["count"] = c + len(embs)
        if name and not entry.get("name"):
            entry["name"] = name
        self.gallery[label] = entry
        self._save_gallery()
        logger.info(f"[Gallery] add {label} (name={entry.get('name','')}), now {len(self.gallery)} speakers.")
        return label

    # ---------- 内部：ASR + 全局指派 ----------

    def _transcribe_with_global_assign(self, file_path: str, language: str = "zh",
                                       hotword_hint: str = "", restrict_gallery_ids=None, **kwargs):

        self._ensure_asr_model()
        
        # 1) ASR + 分句 + 本地说话人（规范化为 SPEAKER_XX），排序修正时间
        segments, full_text = self._run_asr(file_path, hotword_hint)

        # 2) 为每个“本地说话人”抽取嵌入（聚合到 >= min_seconds）
        self._ensure_sv_model()
        local_spk_emb = self._build_local_speaker_embeddings(file_path, segments)

        # 3) 与参考库比对，做全局指派（>阈值指派，否则创建新全局 ID）
        local2global = self._assign_to_gallery(local_spk_emb, threshold=self.match_threshold,
                                               restrict_gallery_ids=restrict_gallery_ids)

        # 4) 回写到 segments，生成各种聚合与行文
        for seg in segments:
            local = seg.get("speaker")
            if local in local2global:
                seg["local_speaker"] = local
                seg["speaker"] = local2global[local]  # 替换成全局ID

        # 给每个片段补充易读的人名（或ID兜底）
        for seg in segments:
            gid = seg.get("speaker")
            seg["speaker_name"] = self._display_name(gid) if gid else ""

        # 以全局ID聚合文本（兼容）
        speaker_texts = defaultdict(list)
        for seg in segments:
            gid = seg.get("speaker")
            if gid:
                speaker_texts[gid].append(seg["text"])
        speaker_texts = {k: "".join(v) for k, v in speaker_texts.items()}

        # 以姓名聚合文本（Name -> 文本）
        name_texts = defaultdict(list)
        for gid, txt in speaker_texts.items():
            name = self._display_name(gid)
            name_texts[name].append(txt)
        name_texts = {k: "".join(v) for k, v in name_texts.items()}

        # 台词行（每段一行：Name: text）
        dialogue_lines = []
        for seg in segments:
            nm = seg.get("speaker_name") or ""
            tx = seg.get("text") or ""
            if nm and tx:
                dialogue_lines.append(f"{nm}: {tx}")
            elif tx:
                dialogue_lines.append(tx)

        result = {
            "text": full_text,
            "segments": segments,
            "speaker_texts": speaker_texts,
            "name_texts": name_texts,
            "dialogue_lines": dialogue_lines,
            "speaker_mapping": {
                "local_to_global": local2global,
                "global_id_to_name": {gid: self._display_name(gid) for gid in speaker_texts.keys()},
                "gallery_size": len(self.gallery),
                "threshold": self.match_threshold,
            },
        }
        return result

    # ---------- ASR 相关 ----------

    def _ensure_asr_model(self):
        if self.funasr_asr is not None:
            return
        logger.info("[FunASR] loading ASR(+VAD+PUNC+SPK) models...")
        self.funasr_asr = AutoModel(
            model=self.asr_model_id,
            vad_model=self.vad_model_id,
            punc_model=self.punc_model_id,
            spk_model=self.spk_model_id,  # 关键：让 sentence_info 带 spk
            disable_update=self.disable_update,
            disable_pbar=self.disable_pbar,
            hub=self.hub,
        )
        logger.info("[FunASR] ASR models ready.")

    def _normalize_local_spk(self, val):
        """
        将 FunASR 的本地说话人标记规范为 'SPEAKER_XX' 字符串。
        允许输入: None, "", 数字/数字字符串, 已是 'SPEAKER_xx' 的字符串。
        """
        if val is None:
            return None
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return None
            up = s.upper()
            if up.startswith("SPEAKER_"):
                return up
            if s.isdigit():
                return f"SPEAKER_{int(s):02d}"
            return s  # 其他字符串，原样返回
        if isinstance(val, (int, float)):
            i = int(val)
            if i < 0:
                return None
            return f"SPEAKER_{i:02d}"
        return f"SPEAKER_{str(val)}"

    def _run_asr(self, audio_path: str, hotword_hint: str = ""):
        gen_kwargs = dict(
            input=os.path.abspath(audio_path),
            batch_size_s=self.batch_size_s,
            disable_pbar=True,
        )
        if hotword_hint:
            gen_kwargs["hotword"] = self._build_hotword(hotword_hint)

        outs = self.funasr_asr.generate(**gen_kwargs)
        if not outs or "sentence_info" not in outs[0]:
            raise RuntimeError("FunASR 返回缺少 sentence_info")

        sentence_infos = outs[0]["sentence_info"]
        segments = []
        full_text_list = []
        for s in sentence_infos:
            text = s.get("text", "")
            st = self._norm_time(s.get("start", 0.0))
            ed = self._norm_time(s.get("end", 0.0))
            # 规范化本地说话人；避免0/1被当作falsy导致跳过
            spk_raw = s.get("spk", None)
            spk = self._normalize_local_spk(spk_raw)
            if spk is None:
                spk = "SPEAKER_00"  # 兜底

            seg = {"text": text, "start_time": st, "end_time": ed, "speaker": spk}
            segments.append(seg)
            full_text_list.append(text)

        # 排序 + 修正反向时间
        segments.sort(key=lambda x: x["start_time"])
        for seg in segments:
            if seg["end_time"] < seg["start_time"]:
                seg["start_time"], seg["end_time"] = seg["end_time"], seg["start_time"]

        # 可选：合并相邻同说话人（更自然的分段）
        segments = self._merge_adjacent_same_speaker(segments)
        return segments, "".join(full_text_list)

    @staticmethod
    def _norm_time(v):
        v = float(v or 0.0)
        return v / 1000.0 if v > 1000 else v

    @staticmethod
    def _merge_adjacent_same_speaker(segments):
        if not segments:
            return segments
        merged = []
        cur = dict(segments[0])
        for s in segments[1:]:
            same = (cur.get("speaker") == s.get("speaker"))
            contig = (s.get("start_time", 0.0) >= cur.get("end_time", 0.0))
            if same and contig:
                cur["text"] = cur.get("text", "") + s.get("text", "")
                cur["end_time"] = max(cur.get("end_time", 0.0), s.get("end_time", 0.0))
            else:
                merged.append(cur)
                cur = dict(s)
        merged.append(cur)
        return merged

    def _build_hotword(self, text: str) -> str:
        if not text:
            return ""
        if JIEBA_OK:
            ws = set()
            for w in jieba.cut(text):
                w = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", w)
                if w:
                    ws.add(w)
            return " ".join(ws)
        text = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ---------- SV / 嵌入提取与匹配 ----------

    def _ensure_sv_model(self):
        if self.funasr_sv is not None:
            return
        logger.info("[FunASR] loading SV model (speaker verification / embedding)...")
        self.funasr_sv = AutoModel(
            model=self.sv_model_id,
            disable_update=self.disable_update,
            disable_pbar=self.disable_pbar,
            hub=self.hub,
            device=self.sv_device,
        )
        logger.info("[FunASR] SV model ready.")

    def _extract_sv_embedding(self, wav_16k_mono_path: str) -> np.ndarray:
        """从 16k 单声道 wav 提取说话人嵌入；兼容 GPU torch.Tensor / list / numpy"""
        outs = self.funasr_sv.generate(input=wav_16k_mono_path, disable_pbar=True)
        if not outs:
            raise RuntimeError("SV model returns empty")
        out0 = outs[0]
        emb = (out0.get("embeddings")
               or out0.get("embedding")
               or out0.get("spk_embedding"))
        if emb is None:
            raise RuntimeError("SV model result has no embedding field")

        emb = _to_numpy_f32(emb)
        if emb.ndim > 1:
            emb = emb.reshape(-1)
        return emb

    def _build_local_speaker_embeddings(self, audio_path: str, segments):
        """
        按“本地说话人标签”聚合时间段，合并至 >= emb_min_seconds 后抽取嵌入并求均值
        返回：{ "SPEAKER_00": np.ndarray(D,), ... }
        """
        abs_path = os.path.abspath(audio_path)
        base_wav = _ensure_wav_16k_mono(abs_path)

        # 收集每个本地说话人的片段
        buckets = defaultdict(list)  # local_spk -> [(st, ed), ...]
        for seg in segments:
            spk = seg.get("speaker")
            if spk is None or spk == "":
                continue
            st, ed = float(seg["start_time"]), float(seg["end_time"])
            if ed > st:
                buckets[spk].append((st, ed))

        # 对每个说话人的片段按时间排序，并聚合成 >= emb_min_seconds 的窗口
        local2emb = {}
        for spk, se_list in buckets.items():
            se_list = sorted(se_list, key=lambda x: x[0])
            windows = []
            cur_st, cur_ed = None, None
            for (st, ed) in se_list:
                if cur_st is None:
                    cur_st, cur_ed = st, ed
                else:
                    # 若间隔 <= emb_max_gap，则拼接
                    if st - cur_ed <= self.emb_max_gap:
                        cur_ed = max(cur_ed, ed)
                    else:
                        windows.append((cur_st, cur_ed))
                        cur_st, cur_ed = st, ed
            if cur_st is not None:
                windows.append((cur_st, cur_ed))

            # 若窗口仍然过短，则相邻窗口合并直到 >= emb_min_seconds
            merged_win = []
            acc_st, acc_ed = None, None
            for (st, ed) in windows:
                if acc_st is None:
                    acc_st, acc_ed = st, ed
                else:
                    if (acc_ed - acc_st) < self.emb_min_seconds:
                        # 合并
                        if st - acc_ed <= self.emb_max_gap:
                            acc_ed = max(acc_ed, ed)
                        else:
                            # 无法再合并，先收下，再开新窗口
                            merged_win.append((acc_st, acc_ed))
                            acc_st, acc_ed = st, ed
                    else:
                        merged_win.append((acc_st, acc_ed))
                        acc_st, acc_ed = st, ed
            if acc_st is not None:
                # 最后一个窗口，若仍短，也照样留下以保证有嵌入
                merged_win.append((acc_st, acc_ed))

            # 抽取嵌入并求均值
            embs = []
            for (st, ed) in merged_win:
                if ed <= st:
                    continue
                slice_wav = _slice_to_wav(base_wav, st, ed)
                emb = self._extract_sv_embedding(slice_wav)
                embs.append(emb)
            if not embs:
                continue
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            local2emb[spk] = mean_emb
        return local2emb

    def _assign_to_gallery(self, local_spk_emb: dict, threshold: float = 0.85, restrict_gallery_ids=None):
        """
        把本地说话人（SPEAKER_00/01/...）指派到参考库里已有的全局ID；
        若最高相似度 < 阈值，则创建新全局ID，写入库并保存。
        可选 restrict_gallery_ids：若提供，仅在该列表中匹配。
        返回：{"SPEAKER_00": "global_spk_1", ...}
        """
        # 候选集合
        if restrict_gallery_ids:
            use_items = [(gid, self.gallery[gid]) for gid in restrict_gallery_ids if gid in self.gallery]
        else:
            use_items = list(self.gallery.items())

        local2global = {}
        for lspk, emb in local_spk_emb.items():
            best_gid, best_sim = None, -1.0
            sims = []
            for gid, entry in use_items:
                ref = np.asarray(entry["embedding"], dtype=np.float32)
                sim = _cosine_sim(emb, ref)
                sims.append((gid, sim))
                if sim > best_sim:
                    best_sim, best_gid = sim, gid

            if self.debug:
                sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"[MatchDebug] {lspk} top3: " + ", ".join([f"{g}:{s:.3f}" for g, s in sims_sorted]))

            if best_gid is not None and best_sim >= threshold:
                local2global[lspk] = best_gid
            else:
                # 创建新全局ID并入库
                new_gid = self._new_global_id()
                self.gallery[new_gid] = {
                    "name": "",
                    "embedding": emb.tolist(),
                    "count": 1,
                }
                self._save_gallery()
                local2global[lspk] = new_gid
                logger.info(f"[Gallery] new speaker {new_gid} created for {lspk} (sim={best_sim:.3f})")
        return local2global

    # ---------- 参考库持久化/工具 ----------

    def _load_gallery(self):
        try:
            if os.path.exists(self.gallery_path):
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    self.gallery = json.load(f)
            else:
                self.gallery = {}
        except Exception as e:
            logger.warning(f"[Gallery] load failed: {e}")
            self.gallery = {}

    def _save_gallery(self):
        try:
            with open(self.gallery_path, "w", encoding="utf-8") as f:
                json.dump(self.gallery, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Gallery] save failed: {e}")

    def _new_global_id(self) -> str:
        # global_spk_1, global_spk_2, ...
        i = 1
        while True:
            gid = f"global_spk_{i}"
            if gid not in self.gallery:
                return gid
            i += 1

    def _probe_duration(self, wav_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", wav_path
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(p.stdout.strip() or 0.0)

    # ---------- 显示名 ----------

    def _display_name(self, gid: str) -> str:
        """把全局说话人ID映射成人名；若没登记name则用ID兜底。"""
        entry = self.gallery.get(gid)
        if entry:
            name = (entry.get("name") or "").strip()
            if name:
                return name
        return gid  # 兜底


# ----------------- 示例 -----------------
if __name__ == "__main__":
    """
    用法示例：
    1) 先登记参考说话人：
        svc = FunASRService()
        g1 = svc.add_reference("旁白.wav", segments=[(0.0, 8.0)], name="旁白")
        g2 = svc.add_reference("老人1.wav", name="老人")

    2) 对新音频转写 + 全局指派（仅在这两位参考中匹配）：
        out = svc.transcribe_audio("meeting.wav", hotword_hint="法院 司法 诉讼 大模型",
                                   restrict_gallery_ids=[g1, g2])
        print(json.dumps(out, ensure_ascii=False, indent=2))
    """
    svc = FunASRService()
    # 这里演示：先登记两个参考
    # g1 = svc.add_reference("/Work21/2023/wangtianrui/codes/my_projects/asr_diarization/旁白.wav", name="旁白")
    # g2 = svc.add_reference("/Work21/2023/wangtianrui/codes/my_projects/asr_diarization/老人1.wav", name="老人")

    # 示例转写（可选限制只在刚登记的两位里匹配）
    out = svc.transcribe_audio(
        "/Users/liyuntao/Downloads/asr_diarization/98-完美世界 第99集 石子陵（喜欢，请订阅，转发，点赞哦！）_chunk1_120850-241360.wav",
        # restrict_gallery_ids=[g1, g2]
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dialogue_lines = out['dialogue_lines']
