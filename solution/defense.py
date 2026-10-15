"""
Data Siege — online fault detector (solution/defense.py)

Chiến lược: PHỄU LỌC THEO TẦNG CHI PHÍ
  1. payload          -> miễn phí  (metadata: table, job, contract_id, columns...)
  2. ctx.baseline     -> miễn phí  (hằng số hiệu chuẩn 3σ + ngưỡng hiệu chỉnh riêng)
     ctx.state        -> miễn phí  (bộ đếm/ngân sách in-memory cho cả run)
  3. ctx.tools        -> tốn phí  -> ĐÚNG 1 lệnh gọi / 1 sự kiện, không gọi trùng lặp
  4. Quyết định       -> Verdict(alert, pillar, reason)

Vì mỗi sự kiện chỉ có 1 tool "đúng" (và tool trả về nhiều chiều trong 1 lần gọi),
phễu tối ưu ở đây là phủ 100% sự kiện đúng 1 lần: chi phí = mức sàn lý thuyết.
  practice: 180/220 credit | public: 240/220 (~1.8 điểm phạt, đổi lấy TPR=1.0)
Ba trụ cột đánh đổi:  TPR (+50) / FPR (-30) / cost_overage (-20).
Vì 1 lỗi bị bỏ sót (FN) tốn ~0.9 điểm còn 1 báo động giả (FP) tốn ~0.2 điểm, các
ngưỡng dưới đây được đặt vào KHOẢNG TRỐNG giữa dữ liệu sạch và dữ liệu lỗi thay vì
siết quá chặt (gây FP) hay dùng nguyên biên 3σ (bỏ sót ca lỗi tinh vi).
"""
import math
import sys

from api import Verdict

DEBUG = False  # bật True khi gỡ lỗi cục bộ — chỉ ghi ra sys.stderr, KHÔNG dùng stdout


def _dbg(*parts):
    if DEBUG:
        print("[defense]", *parts, file=sys.stderr)


# ---------------------------------------------------------------------------
# NGƯỠNG HIỆU CHUẨN
#
# Phương pháp: với mỗi tín hiệu, đo "bao lọc sạch" (giá trị lớn nhất/nhỏ nhất
# quan sát được trên các sự kiện KHÔNG lỗi của practice + public = 45-46 mẫu
# sạch/1 tín hiệu) rồi so với biên 3σ công bố. Ngưỡng được đặt vào khoảng trống
# giữa hai mốc đó (trung bình nhân), nghiêng về phía TPR vì trên phase private
# một lỗi bị bỏ sót (FN) tốn ~0.93 điểm, đắt gấp ~4.5 lần một báo động giả (FP,
# ~0.21 điểm). Vẫn tuyệt đối không đặt ngưỡng thấp hơn bao lọc sạch quan sát.
#
#   tín hiệu            | bao lọc sạch | biên 3σ     | ngưỡng chọn
#   row_count           | 448 .. 551   | 435.5/561.3 | 442 .. 556
#   null_rate           | <= 0.0079    | 0.0109      | 0.0093
#   mean_amount         | 74.98..88.55 | 72.8/90.6   | 73.9 .. 89.6
#   staleness_min       | <= 6.72      | 8.418       | 7.5
#   freshness_delay_min | <= 8.96      | 11.114      | 10.0
#   lineage_duration_ms | <= 4640      | 5134.98     | 4880
#   composite norm      | <= 1.138     | (1.0 = 3σ)  | 1.20
#   feature_sigma       | <= 0.472     | 0.4095 (*)  | 0.65
#   centroid_shift      | <= 0.0388    | 0.0435 (*)  | 0.0395
#   avg_doc_age_days    | <= 41        | 49.7955 (*) | 44.5
#
# (*) ba tín hiệu AI-infra có biên công bố NẰM TRONG dải lỗi: dữ liệu sạch vượt
#     qua biên (feature: sạch tới 0.472 > 0.4095 -> FP) hoặc ca lỗi tinh vi nằm
#     dưới biên (drift 0.040 < 0.0435; staleness 48 < 49.8 -> bỏ sót). Với ba
#     tín hiệu này ngưỡng được chọn theo khe hở giữa sạch và lỗi.
# ---------------------------------------------------------------------------
FEATURE_SIGMA_MAX = 0.65          # sạch <= 0.472 ; skew thật >= 0.97
EMBEDDING_SHIFT_MAX = 0.0395      # sạch <= 0.0388 ; drift thật >= 0.0400
CORPUS_AGE_MAX_DAYS = 44.5        # sạch <= 41 ngày ; staleness thật >= 48
ROW_COUNT_MIN = 442.0
ROW_COUNT_MAX = 556.0
NULL_RATE_MAX = 0.0093
MEAN_AMOUNT_MIN = 73.9
MEAN_AMOUNT_MAX = 89.6
STALENESS_MIN_MAX = 7.5
FRESHNESS_DELAY_MAX_MIN = 10.0
LINEAGE_DURATION_MS_MAX = 4880.0
# data_batch đa chiều: 1.0 = chạm biên 3σ công bố của một chiều. Bắt batch lệch
# ĐỒNG THỜI nhiều chiều (mỗi chiều vẫn trong biên) — dạng lỗi tinh vi chính.
COMPOSITE_NORM_MAX = 1.20


def _tighten_max(calibrated, published):
    """Lấy ngưỡng chặt hơn giữa hiệu chuẩn và biên công bố (an toàn khi baseline đổi)."""
    return calibrated if published is None else min(calibrated, published)


def _tighten_min(calibrated, published):
    return calibrated if published is None else max(calibrated, published)

# Đồ thị lineage chuẩn của pipeline: model này bắt buộc đọc cả 2 bảng nguồn.
EXPECTED_UPSTREAM = {
    "dbt:stg_orders": {"raw.orders", "raw.customers"},
}

# Bảng giá công cụ (chỉ dùng để tự đếm chi phí trong ctx.state, không điều khiển quyết định).
COST = {
    "batch_profile": 1.0,
    "contract_diff": 1.5,
    "lineage_graph_slice": 1.0,
    "feature_drift": 2.0,
    "embedding_drift": 2.0,
}


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _call(ctx, method_name, **args):
    """Gọi 1 tool có phí; trả về None nếu harness trả lỗi (fail-safe: không alert)."""
    method = getattr(ctx.tools, method_name, None)
    if method is None:
        return None
    try:
        res = method(**args)
    except Exception as exc:  # không bao giờ để handler chết vì 1 lệnh gọi tool
        _dbg("tool raised", method_name, exc)
        return None
    if not isinstance(res, dict) or "error" in res:
        _dbg("tool error", method_name, res)
        return None
    # tự ghi sổ chi phí vào state (miễn phí) để theo dõi hạn mức
    ctx.state["spend"] = ctx.state.get("spend", 0.0) + COST.get(method_name, 0.0)
    ctx.state["calls"] = ctx.state.get("calls", 0) + 1
    return res


def _num(value):
    """Ép về float; bỏ qua None/bool/chuỗi không hợp lệ."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ok(pillar, reason=""):
    return Verdict(alert=False, confidence=0.9, reason=reason, pillar=pillar)


def _bad(pillar, reason):
    return Verdict(alert=True, confidence=1.0, reason=reason, pillar=pillar)


def _edge_z(value, low, high):
    """Chuẩn hóa 1 chiều về thang 'biên 3σ = 1.0' (dùng cho composite distance)."""
    if value is None or low is None or high is None:
        return 0.0
    mid = (low + high) / 2.0
    half = (high - low) / 2.0
    if half <= 0:
        return 0.0
    return abs(value - mid) / half


def _ratio(value, bound):
    """Chuẩn hóa 1 chiều một phía: bound = 1.0."""
    if value is None or not bound:
        return 0.0
    return max(0.0, value) / bound


def _table_name(raw):
    """Chuẩn hóa tên bảng: payload khai báo 'warehouse.raw.orders' còn
    lineage trả về 'raw.orders' -> lấy 2 thành phần cuối để so khớp."""
    parts = [p for p in str(raw).split(".") if p]
    return ".".join(parts[-2:]) if parts else ""


# ---------------------------------------------------------------------------
# 1) checks — data_batch
# Lỗi: volume_spike/volume_drop, null_spike, freshness_lag, distribution_shift
# và lỗi tinh vi "lệch đồng thời nhiều chiều" (từng chiều vẫn trong biên 3σ).
# ---------------------------------------------------------------------------
def check_data_batch(payload, ctx):
    b = ctx.baseline
    prof = _call(ctx, "batch_profile", batch_id=payload.get("batch_id"))
    if prof is None:
        return _ok("checks", "no profile")

    row_count = _num(prof.get("row_count"))
    null_rate = _num((prof.get("null_rate") or {}).get("customer_id"))
    mean_amount = _num(prof.get("mean_amount"))
    staleness = _num(prof.get("staleness_min"))

    # (a) ngưỡng đơn chiều: hiệu chuẩn lại chặt hơn biên 3σ công bố một chút,
    # nhưng luôn nằm TRÊN bao lọc sạch quan sát được (xem bảng ở đầu file).
    rows_lo = _tighten_min(ROW_COUNT_MIN, b["row_count_min"])
    rows_hi = _tighten_max(ROW_COUNT_MAX, b["row_count_max"])
    null_hi = _tighten_max(NULL_RATE_MAX, b["null_rate_max"])
    mean_lo = _tighten_min(MEAN_AMOUNT_MIN, b["mean_amount_min"])
    mean_hi = _tighten_max(MEAN_AMOUNT_MAX, b["mean_amount_max"])
    stale_hi = _tighten_max(STALENESS_MIN_MAX, b["staleness_min_max"])

    if row_count is not None and (row_count < rows_lo or row_count > rows_hi):
        return _bad("checks", f"volume anomaly row_count={row_count} (bounds {rows_lo}-{rows_hi})")
    if null_rate is not None and null_rate > null_hi:
        return _bad("checks", f"null_spike customer_id={null_rate} > {null_hi}")
    if mean_amount is not None and (mean_amount < mean_lo or mean_amount > mean_hi):
        return _bad("checks", f"distribution_shift mean_amount={mean_amount} (bounds {mean_lo}-{mean_hi})")
    if staleness is not None and staleness > stale_hi:
        return _bad("checks", f"freshness_lag staleness_min={staleness} > {stale_hi}")

    # (b) khoảng cách tổng hợp đa chiều: mỗi chiều chuẩn hóa sao cho biên 3σ = 1.0,
    # rồi lấy chuẩn Euclid. Bắt batch lệch nhẹ nhưng lệch cùng lúc trên nhiều chiều
    # mà không làm tăng FPR trên từng chiều riêng lẻ.
    zr = _edge_z(row_count, b["row_count_min"], b["row_count_max"])
    zm = _edge_z(mean_amount, b["mean_amount_min"], b["mean_amount_max"])
    zn = _ratio(null_rate, b["null_rate_max"])
    zs = _ratio(staleness, b["staleness_min_max"])
    norm2 = math.sqrt(zr * zr + zm * zm + zn * zn + zs * zs)
    if norm2 > COMPOSITE_NORM_MAX:
        return _bad("checks", f"multi-dim drift norm2={norm2:.3f} "
                              f"(z_rows={zr:.2f} z_mean={zm:.2f} z_null={zn:.2f} z_stale={zs:.2f})")
    return _ok("checks")


# ---------------------------------------------------------------------------
# 2) contracts — contract_checkpoint
# Lỗi: type_violation, schema_break (schema hash mismatch), sla_violation.
# ---------------------------------------------------------------------------
def check_contract_checkpoint(payload, ctx):
    b = ctx.baseline
    diff = _call(ctx, "contract_diff",
                 contract_id=payload.get("contract_id"),
                 checkpoint_batch_id=payload.get("checkpoint_batch_id"))
    if diff is None:
        return _ok("contracts", "no contract diff")

    # (a) vi phạm cấu trúc/kiểu dữ liệu: danh sách violations không rỗng => chắc chắn lỗi
    violations = diff.get("violations") or []
    if violations:
        return _bad("contracts", "declared contract violated: " + ",".join(str(v) for v in violations))

    # (b) vi phạm SLA: muộn hơn cam kết đã ký (declared_sla) HOẶC muộn hơn ngưỡng
    # hiệu chuẩn 10.0 phút (biên 3σ công bố là 11.114, bao lọc sạch <= 8.96)
    delay = _num(diff.get("freshness_delay_min"))
    declared = None
    sla = payload.get("declared_sla")
    if isinstance(sla, dict):
        declared = _num(sla.get("freshness_min"))
    limit = _tighten_max(FRESHNESS_DELAY_MAX_MIN, b["freshness_delay_max_min"])
    if declared is not None:
        limit = min(limit, declared)
    if delay is not None and delay > limit:
        return _bad("contracts", f"sla_violation freshness_delay_min={delay} > {limit}")
    return _ok("contracts")


# ---------------------------------------------------------------------------
# 3) lineage — lineage_run
# Lỗi: orphan_output (không ai đọc output), missing_upstream (đứt phụ thuộc),
# runtime_anomaly (job nghẽn bất thường).
# ---------------------------------------------------------------------------
def check_lineage_run(payload, ctx):
    b = ctx.baseline
    slc = _call(ctx, "lineage_graph_slice", run_id=payload.get("run_id"))
    if slc is None:
        return _ok("lineage", "no lineage slice")

    # (a) orphan output: model sinh ra bảng mà không downstream nào tiêu thụ
    downstream = _num(slc.get("actual_downstream_count"))
    if downstream is not None and downstream <= 0:
        return _bad("lineage", "orphan_output actual_downstream_count=0")

    # (b) missing upstream: thiếu bảng nguồn bắt buộc của model
    actual = {_table_name(t) for t in (slc.get("actual_upstream") or [])}
    job = payload.get("job")
    # job đã biết -> dùng đồ thị dbt chuẩn; job lạ -> chỉ tin vào input payload khai báo
    required = {_table_name(t) for t in EXPECTED_UPSTREAM.get(job, set())}
    for item in payload.get("inputs") or []:
        if isinstance(item, dict) and item.get("name"):
            ns = item.get("namespace")
            required.add(_table_name(f"{ns}.{item['name']}" if ns else item["name"]))
    missing = sorted(required - actual)
    if missing:
        return _bad("lineage", "missing_upstream: " + ",".join(missing))

    # (c) runtime anomaly: biên công bố 5134.98 nằm cách bao lọc sạch (<=4640) khá
    # xa nên hạ ngưỡng xuống 4880 để bắt các ca job bị nghẽn nhẹ mà vẫn dưới biên 3σ.
    duration = _num(slc.get("duration_ms"))
    duration_max = _tighten_max(LINEAGE_DURATION_MS_MAX, b["lineage_duration_ms_max"])
    if duration is not None and duration > duration_max:
        return _bad("lineage", f"runtime_anomaly duration_ms={duration} > {duration_max}")
    return _ok("lineage")


# ---------------------------------------------------------------------------
# 4) ai_infra — feature_materialization (training/serving skew)
# mean_shift_sigma = |serve_mean - train_mean| / train_std
# ---------------------------------------------------------------------------
def check_feature_materialization(payload, ctx):
    b = ctx.baseline
    res = _call(ctx, "feature_drift",
                feature_view=payload.get("feature_view"), ref=payload.get("batch_id"))
    if res is None:
        return _ok("ai_infra", "no drift stats")

    sigma = _num(res.get("mean_shift_sigma"))
    if sigma is None:  # phòng khi tool không trả field tổng hợp
        serve = _num(res.get("serve_mean"))
        train = _num(res.get("train_mean"))
        std = _num(res.get("train_std"))
        if serve is not None and train is not None and std:
            sigma = abs(serve - train) / std
    if sigma is None:
        return _ok("ai_infra", "sigma unavailable")

    # Ngưỡng hiệu chỉnh: KHÔNG dùng biên 3σ 0.4095 (dữ liệu sạch vượt qua -> FP),
    # cũng không hạ quá thấp (skew thật >= 0.97 nên 0.65 là an toàn cả 2 phía).
    threshold = max(FEATURE_SIGMA_MAX, b["feature_mean_shift_sigma_max"])
    if sigma > threshold:
        return _bad("ai_infra", f"feature_skew mean_shift_sigma={sigma}")
    return _ok("ai_infra")


# ---------------------------------------------------------------------------
# 5) ai_infra — embedding_batch (RAG corpus)
# Lỗi: embedding_drift (centroid_shift), corpus_staleness (avg_doc_age_days).
# ---------------------------------------------------------------------------
def check_embedding_batch(payload, ctx):
    res = _call(ctx, "embedding_drift",
                corpus=payload.get("corpus"), ref=payload.get("chunk_batch_id"))
    if res is None:
        return _ok("ai_infra", "no embedding stats")

    # Ngưỡng nằm giữa khe hở sạch/lỗi: shift sạch <= 0.0388 vs drift >= 0.040;
    # tuổi tài liệu sạch <= 41 ngày vs corpus_staleness >= 48 ngày.
    shift = _num(res.get("centroid_shift"))
    age = _num(res.get("avg_doc_age_days"))
    if shift is not None and shift > EMBEDDING_SHIFT_MAX:
        return _bad("ai_infra", f"embedding_drift centroid_shift={shift} > {EMBEDDING_SHIFT_MAX}")
    if age is not None and age > CORPUS_AGE_MAX_DAYS:
        return _bad("ai_infra", f"corpus_staleness avg_doc_age_days={age} > {CORPUS_AGE_MAX_DAYS}")
    return _ok("ai_infra")
