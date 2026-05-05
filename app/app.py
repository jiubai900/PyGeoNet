from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Project path setup
# -----------------------------------------------------------------------------
# Expected layout:
#   pygeonet-2/
#       download.py / normalize.py / model.py / standard.py / search.py ...
#       utils/
#       app/
#           app.py
#
# Only this app/ folder is added/changed. The original PyGeoNet function package
# remains untouched and is imported from PROJECT_ROOT.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def prefer_project_root() -> None:
    """Keep the original PyGeoNet root before app/ on sys.path.

    This is important because PyGeoNet has a top-level package named ``utils``.
    If the Flask app is started with ``python app/app.py``, Python may place
    ``app/`` before the project root and accidentally import ``app/utils`` as
    top-level ``utils``. That causes errors such as:
        No module named 'utils.search'
    """
    root = str(PROJECT_ROOT)
    sys.path[:] = [p for p in sys.path if str(Path(p or os.getcwd()).resolve()) != root]
    sys.path.insert(0, root)


def purge_wrong_utils_package() -> None:
    """Remove a shadowed top-level utils package if it was imported from app/."""
    utils_mod = sys.modules.get("utils")
    if utils_mod is None:
        return

    module_file = getattr(utils_mod, "__file__", None)
    if not module_file:
        return

    project_utils = (PROJECT_ROOT / "utils").resolve()
    try:
        Path(module_file).resolve().relative_to(project_utils)
        return
    except ValueError:
        pass

    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]


prefer_project_root()
purge_wrong_utils_package()

RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
CACHE_DIR = RUNTIME_DIR / "cache"
GENE_LIST_PATH = PROJECT_ROOT / "utils" / "model" / "gene_list.csv"

for folder in (RUNTIME_DIR, UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(BASE_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    return app


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["DEBUG"] = True
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB upload cap; adjust if needed.
app.secret_key = os.environ.get("PYGEONET_APP_SECRET", "pygeonet-demo-secret-key")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def import_project_module(module_name: str):
    """Import a module from the original PyGeoNet root, not from app/."""
    prefer_project_root()
    purge_wrong_utils_package()

    loaded = sys.modules.get(module_name)
    if loaded is not None:
        module_file = getattr(loaded, "__file__", None)
        if module_file:
            try:
                Path(module_file).resolve().relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                del sys.modules[module_name]

    return importlib.import_module(module_name)


def load_function(module_name: str, function_name: str) -> Callable:
    """Load a function/class from the original PyGeoNet project root lazily."""
    module = import_project_module(module_name)
    return getattr(module, function_name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_job_dir(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = OUTPUT_DIR / f"{prefix}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def make_zip(source_dir: Path, zip_name: Optional[str] = None) -> Path:
    source_dir = Path(source_dir)
    zip_base = source_dir.parent / (zip_name or source_dir.name)
    zip_file = shutil.make_archive(str(zip_base), "zip", str(source_dir))
    return Path(zip_file)


def save_uploaded_file(file_storage, target_dir: Path) -> Path:
    ensure_dir(target_dir)
    filename = secure_filename(file_storage.filename)
    if not filename:
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    saved_path = target_dir / filename
    file_storage.save(saved_path)
    return saved_path


def _safe_extract_member(zip_ref: zipfile.ZipFile, member: zipfile.ZipInfo, target_dir: Path, flatten: bool) -> None:
    if member.is_dir():
        return
    member_name = Path(member.filename).name if flatten else member.filename
    if not member_name:
        return
    output_path = (target_dir / member_name).resolve()
    target_root = target_dir.resolve()
    if not str(output_path).startswith(str(target_root)):
        raise ValueError(f"Unsafe zip member path: {member.filename}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(zip_ref.read(member))


def extract_zip(zip_path: Path, target_dir: Path, *, flatten: bool = False, keep_zip: bool = False) -> Path:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            _safe_extract_member(zip_ref, member, target_dir, flatten=flatten)
    if not keep_zip and zip_path.exists():
        zip_path.unlink()
    return target_dir


def unwrap_single_dir(path: Path) -> Path:
    """Return the only child directory when a zip extracts to one top-level folder."""
    path = Path(path)
    if not path.is_dir():
        return path
    children = [p for p in path.iterdir() if not p.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return path


def prepare_uploaded_data(
    file_storage,
    target_dir: Path,
    *,
    flatten_zip: bool = False,
    unwrap_root: bool = False,
) -> Path:
    saved_path = save_uploaded_file(file_storage, target_dir)
    if zipfile.is_zipfile(saved_path):
        extract_dir = target_dir / saved_path.stem
        reset_dir(extract_dir)
        extract_zip(saved_path, extract_dir, flatten=flatten_zip)
        return unwrap_single_dir(extract_dir) if unwrap_root else extract_dir
    return saved_path


def first_matching_file(folder: Path, suffixes: Iterable[str]) -> Optional[Path]:
    suffixes = {s.lower() for s in suffixes}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def list_matching_files(folder: Path, suffixes: Iterable[str]) -> list[Path]:
    suffixes = {s.lower() for s in suffixes}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]


def send_zip(zip_path: Path, download_name: Optional[str] = None):
    return send_file(str(zip_path), as_attachment=True, download_name=download_name or zip_path.name)


def parse_int_list(value: str, default: Sequence[int]) -> list[int]:
    values: list[int] = []
    for item in value.replace("，", ",").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or list(default)


def parse_seed_list(value: str, default: Sequence[int]) -> list[int]:
    value = (value or "").replace("，", ",").replace(";", ",").replace(" ", ",")
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    return seeds or list(default)


def write_run_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


@app.errorhandler(Exception)
def handle_exception(exc: Exception):
    logger.exception("Unhandled app error")
    detail = traceback.format_exc()
    if app.debug:
        return f"平台运行出错：{exc}\n\n{detail}", 500, {"Content-Type": "text/plain; charset=utf-8"}
    return "平台运行出错，请查看 app/app.log。", 500


# -----------------------------------------------------------------------------
# Request handlers
# -----------------------------------------------------------------------------
def handle_download_request():
    download_all = load_function("download", "all")
    down_again = load_function("download", "down_again")  # loaded here to validate module; route uses it separately.
    gpl = load_function("download", "gpl")
    matrix = load_function("download", "matrix")
    soft = load_function("download", "soft")
    suppl = load_function("download", "suppl")
    _ = down_again

    job_dir = create_job_dir("download")
    save_path = ensure_dir(job_dir / "data")
    log_path = ensure_dir(job_dir / "log")
    input_path = ensure_dir(job_dir / "input")

    gse_source = None
    upload = request.files.get("file")
    if upload and upload.filename:
        gse_source = str(save_uploaded_file(upload, input_path))
    elif request.form.get("text"):
        gse_source = request.form.get("text", "").strip()

    if not gse_source:
        return "没有可处理的数据", 400

    options = request.form.getlist("options")
    if not options:
        return "请至少选择一种下载类型", 400

    if "all" in options:
        download_all(gse_source, str(save_path), str(log_path))
    else:
        if "soft" in options:
            soft(gse_source, str(save_path), str(log_path))
        if "matrix" in options:
            matrix(gse_source, str(save_path), str(log_path))
        if "suppl" in options:
            suppl(gse_source, str(save_path), str(log_path))
        if "gpl" in options:
            gpl(gse_source, str(save_path), str(log_path))

    zip_path = make_zip(job_dir, job_dir.name)
    return send_zip(zip_path)


def handle_standard_request():
    # Import the original normalize module from the PyGeoNet root.
    # Some historical copies of normalize.py use as_completed() without importing it;
    # this runtime patch keeps the app usable without editing the function package file.
    import concurrent.futures

    normalize_mod = import_project_module("normalize")
    if not hasattr(normalize_mod, "as_completed"):
        normalize_mod.as_completed = concurrent.futures.as_completed
    get_data = normalize_mod.get_data

    job_dir = create_job_dir("standard")
    raw_dir = ensure_dir(job_dir / "raw_data")
    gpl_dir = ensure_dir(job_dir / "gpl_data")
    output_dir = ensure_dir(job_dir / "result")

    single = request.form.get("single") == "on"

    data_file = request.files.get("file-data")
    gpl_file = request.files.get("file-plg")

    # Compatible with older forms that submit both files under name="file".
    if data_file is None and gpl_file is None:
        file_list = [f for f in request.files.getlist("file") if f and f.filename]
        if len(file_list) >= 1:
            data_file = file_list[0]
        if len(file_list) >= 2:
            gpl_file = file_list[1]

    if data_file is None or not data_file.filename:
        return "请上传需要规范化的数据文件", 400

    data_input = prepare_uploaded_data(data_file, raw_dir, flatten_zip=False, unwrap_root=single)
    gpl_input = None
    if gpl_file and gpl_file.filename:
        # GPL data can be a directory or one file. Preserve tree to avoid losing platform files.
        gpl_input = prepare_uploaded_data(gpl_file, gpl_dir, flatten_zip=False, unwrap_root=True)

    if gpl_input:
        get_data(str(data_input), str(gpl_input), str(output_dir), single_operation=single)
    else:
        get_data(str(data_input), save_path=str(output_dir), single_operation=single)

    zip_path = make_zip(output_dir, "standard_result")
    return send_zip(zip_path)


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
@app.route("/")
def main():
    return render_template("index/main.html", active_page="home")


@app.route("/search")
def search_page():
    return render_template("search.html", active_page="search")


@app.route("/download")
def download_page():
    return render_template("download.html", active_page="download")


@app.route("/standardization")
def standardization_page():
    return render_template("standardization.html", active_page="standardization")


@app.route("/plot")
def analysis_page():
    return render_template("plot.html", active_page="analysis")


@app.route("/models")
def models_page():
    return render_template("models.html", active_page="models")


@app.route("/super-model")
def super_model_page():
    return render_template("super-model.html", active_page="super-model")


# -----------------------------------------------------------------------------
# GEO search and download
# -----------------------------------------------------------------------------
@app.route("/search-geo", methods=["POST"])
def search_geo():
    contents = load_function("search", "contents")
    gse = load_function("search", "gse")

    form_keys = [
        "ALL", "AUTH", "GTYP", "DESC", "ETYP", "FILT", "ACCN", "MESH", "NPRO", "NSAM",
        "ORGN", "PTYP", "PRO", "PDAT", "RGPL", "RGSE", "GEID", "SRC", "STYP", "VTYP",
        "INST", "SSDE", "SSTP", "SFIL", "TAGL", "TITL", "UDAT", "retmax",
    ]
    kwargs = {key: request.form.get(key) for key in form_keys}
    result = gse(**kwargs)
    if not result:
        return "未检索到符合条件的 GEO 数据集", 400

    job_dir = create_job_dir("search")
    contents(result, save_path=str(job_dir), output_format="xlsx")
    excel_path = job_dir / "GEO.xlsx"
    return send_file(str(excel_path), as_attachment=True, download_name="GEO.xlsx")


@app.route("/down-file", methods=["POST"])
def submit_down():
    # Compatible with older standardization forms that incorrectly posted to /down-file.
    if request.form.get("single") is not None or len(request.files.getlist("file")) >= 2:
        return handle_standard_request()
    return handle_download_request()


@app.route("/down-again", methods=["POST"])
def again_down_route():
    down_again = load_function("download", "down_again")

    job_dir = create_job_dir("redownload")
    save_dir = ensure_dir(job_dir / "data")
    log_dir = ensure_dir(job_dir / "log_input")

    upload = request.files.get("file-right")
    if upload is None or not upload.filename:
        return "请上传日志文件或日志压缩包", 400

    saved_path = save_uploaded_file(upload, log_dir)
    log_files: list[Path]

    if zipfile.is_zipfile(saved_path):
        extracted_dir = log_dir / "unzipped"
        reset_dir(extracted_dir)
        extract_zip(saved_path, extracted_dir, flatten=True)
        log_files = list_matching_files(extracted_dir, {".txt"})
    else:
        log_files = [saved_path]

    if not log_files:
        return "压缩包中未找到可用的 txt 日志文件", 400

    for log_file in log_files:
        down_again(str(log_file), str(save_dir))

    zip_path = make_zip(job_dir, job_dir.name)
    return send_zip(zip_path)


# -----------------------------------------------------------------------------
# Normalization and analysis
# -----------------------------------------------------------------------------
@app.route("/standard", methods=["POST"])
def standard_route():
    return handle_standard_request()


@app.route("/get-plot", methods=["POST"])
def get_plot():
    from app.plot import box_plot, clustering, enrichment_plot, heatmap, volcano_plot

    analysis_type = request.form.get("analysis-type")
    data_file = request.files.get("file-input")
    gene_id = request.form.get("gene-id", "").strip()
    cluster_count = request.form.get("cluster-count", "").strip()

    if data_file is None or not data_file.filename:
        return "请上传分析数据文件", 400

    job_dir = create_job_dir("plot")
    input_dir = ensure_dir(job_dir / "input")
    output_dir = ensure_dir(job_dir / "result")
    data_path = save_uploaded_file(data_file, input_dir)

    if analysis_type == "volcano":
        volcano_plot(str(data_path), str(output_dir))
    elif analysis_type == "boxplot":
        if not gene_id:
            return "箱线图分析需要输入基因 ID", 400
        box_plot(str(data_path), gene_id, str(output_dir))
    elif analysis_type == "heatmap":
        heatmap(str(data_path), str(output_dir))
    elif analysis_type == "clustering":
        n_clusters = int(cluster_count) if cluster_count else 3
        clustering(str(data_path), n_clusters=n_clusters, save_path=str(output_dir))
    elif analysis_type == "enrichment":
        enrichment_plot(str(data_path), str(output_dir))
    else:
        return "不支持的分析类型", 400

    zip_path = make_zip(output_dir, "plot_result")
    return send_zip(zip_path)


@app.route("/pca-feature", methods=["POST"])
def pca_feature_route():
    pca = load_function("standard", "pca")

    data_file = request.files.get("pca_file_data")
    n_components = int(request.form.get("n_components", 3))

    if data_file is None or not data_file.filename:
        return "请上传主成分分析数据文件", 400

    job_dir = create_job_dir("pca")
    input_dir = ensure_dir(job_dir / "input")
    png_dir = ensure_dir(job_dir / "png")
    output_dir = ensure_dir(job_dir / "result")

    prepared_input = prepare_uploaded_data(data_file, input_dir, flatten_zip=False, unwrap_root=True)
    pca(str(prepared_input), str(png_dir), str(output_dir), n_components=n_components)

    zip_path = make_zip(job_dir, job_dir.name)
    return send_zip(zip_path)


# -----------------------------------------------------------------------------
# Classification model
# -----------------------------------------------------------------------------
def model_symbols():
    model_mod = import_project_module("model")
    return {
        "GAEModel": getattr(model_mod, "GAEModel"),
        "GATModel": getattr(model_mod, "GATModel"),
        "GCNModel": getattr(model_mod, "GCNModel"),
        "bce_loss": getattr(model_mod, "bce_loss"),
        "mse_loss": getattr(model_mod, "mse_loss"),
        "class_train": getattr(model_mod, "class_train"),
        "class_predict": getattr(model_mod, "class_predict"),
        "edge_pre": getattr(model_mod, "edge_pre"),
    }


@app.route("/train", methods=["POST"])
def train_model():
    symbols = model_symbols()

    adj_matrix_file = request.files.get("adj_matrix")
    feature_matrix_file = request.files.get("feature_matrix")

    if not adj_matrix_file or not adj_matrix_file.filename:
        return "请上传邻接矩阵文件", 400
    if not feature_matrix_file or not feature_matrix_file.filename:
        return "请上传特征矩阵文件", 400

    num_classes = int(request.form.get("class_count", 2))
    epochs = int(request.form.get("iterations", 200))
    learning_rate = float(request.form.get("learning_rate", 0.01))
    model_selection = request.form.get("model_selection", "GAE")
    loss_function = request.form.get("loss_function", "bce")

    job_dir = create_job_dir("train")
    input_dir = ensure_dir(job_dir / "input")
    output_dir = ensure_dir(job_dir / "result")

    adj_matrix_path = save_uploaded_file(adj_matrix_file, input_dir)
    feature_matrix_path = save_uploaded_file(feature_matrix_file, input_dir)

    model_map = {
        "GAE": symbols["GAEModel"],
        "GAT": symbols["GATModel"],
        "GCN": symbols["GCNModel"],
    }
    loss_map = {"bce": symbols["bce_loss"], "mse": symbols["mse_loss"]}

    symbols["class_train"](
        str(adj_matrix_path),
        str(feature_matrix_path),
        num_classes,
        str(output_dir),
        epochs,
        learning_rate,
        model_map.get(model_selection, symbols["GAEModel"]),
        loss_map.get(loss_function, symbols["bce_loss"]),
    )

    zip_path = make_zip(output_dir, "train_result")
    return send_zip(zip_path)


@app.route("/predict", methods=["POST"])
def predict_model():
    symbols = model_symbols()

    adj_matrix_file = request.files.get("adj_matrix")
    feature_matrix_file = request.files.get("feature_matrix")
    model_parameters_file = request.files.get("model_parameters")

    if not adj_matrix_file or not adj_matrix_file.filename:
        return "请上传邻接矩阵文件", 400
    if not feature_matrix_file or not feature_matrix_file.filename:
        return "请上传特征矩阵文件", 400
    if not model_parameters_file or not model_parameters_file.filename:
        return "请上传模型参数文件", 400

    model_selection = request.form.get("model_selection", "GAE")
    num_classes = int(request.form.get("class_count", 2))

    job_dir = create_job_dir("predict")
    input_dir = ensure_dir(job_dir / "input")
    output_dir = ensure_dir(job_dir / "result")

    adj_matrix_path = save_uploaded_file(adj_matrix_file, input_dir)
    feature_matrix_path = save_uploaded_file(feature_matrix_file, input_dir)
    model_upload_path = save_uploaded_file(model_parameters_file, input_dir)

    if zipfile.is_zipfile(model_upload_path):
        model_extract_dir = input_dir / "model_files"
        reset_dir(model_extract_dir)
        extract_zip(model_upload_path, model_extract_dir, flatten=False)
        model_path = first_matching_file(model_extract_dir, {".pth", ".pt", ".pkl", ".ckpt"})
        if model_path is None:
            return "模型压缩包中未找到可用的参数文件", 400
    else:
        model_path = model_upload_path

    model_map = {
        "GAE": symbols["GAEModel"],
        "GAT": symbols["GATModel"],
        "GCN": symbols["GCNModel"],
    }
    symbols["class_predict"](
        str(adj_matrix_path),
        str(feature_matrix_path),
        str(model_path),
        num_classes,
        str(output_dir),
        model_map.get(model_selection, symbols["GAEModel"]),
    )

    zip_path = make_zip(output_dir, "predict_result")
    return send_zip(zip_path)


# -----------------------------------------------------------------------------
# Relationship prediction workflow
# -----------------------------------------------------------------------------
@app.route("/similarity", methods=["POST"])
def similarity_route():
    similarity = load_function("standard", "similarity")

    sim_file = request.files.get("sim_file-input")
    if sim_file is None or not sim_file.filename:
        return "请上传原始表达数据压缩包", 400

    sim_th_raw = request.form.get("sim_th", "").strip()
    sim_th = float(sim_th_raw) if sim_th_raw else 0.3

    job_dir = create_job_dir("similarity")
    input_dir = ensure_dir(job_dir / "input")
    output_dir = ensure_dir(job_dir / "result")

    prepared_input = prepare_uploaded_data(sim_file, input_dir, flatten_zip=False, unwrap_root=True)
    similarity(str(prepared_input), str(output_dir), sim_threshold=sim_th)

    zip_path = make_zip(output_dir, "similarity_result")
    return send_zip(zip_path)


@app.route("/statistic", methods=["POST"])
def statistic_route():
    statistic = load_function("standard", "statistic")

    sim_file = request.files.get("sim_file-input")
    sta_file = request.files.get("sta_file-input")

    if sim_file is None or not sim_file.filename:
        return "请上传相似性结果压缩包", 400
    if sta_file is None or not sta_file.filename:
        return "请上传原始表达数据压缩包", 400

    count_th_raw = request.form.get("con_th", "").strip()
    count_th = float(count_th_raw) if count_th_raw else 0.8

    job_dir = create_job_dir("statistic")
    sim_dir = ensure_dir(job_dir / "sim_input")
    expr_dir = ensure_dir(job_dir / "expr_input")
    output_dir = ensure_dir(job_dir / "result")

    prepared_sim = prepare_uploaded_data(sim_file, sim_dir, flatten_zip=False, unwrap_root=True)
    prepared_expr = prepare_uploaded_data(sta_file, expr_dir, flatten_zip=False, unwrap_root=True)

    statistic(str(prepared_sim), str(prepared_expr), str(output_dir), count_threshold=count_th)

    zip_path = make_zip(output_dir, "statistic_result")
    return send_zip(zip_path)


@app.route("/edge", methods=["POST"])
def edge_route():
    symbols = model_symbols()
    edge_pre = symbols["edge_pre"]

    adj_file = request.files.get("adj_file-input")
    gene_file = request.files.get("expr_file-input")
    gene_map_file = request.files.get("map_file-input")

    if adj_file is None or not adj_file.filename:
        return "请上传邻接矩阵文件", 400
    if gene_file is None or not gene_file.filename:
        return "请上传表达数据压缩包", 400
    if gene_map_file is None or not gene_map_file.filename:
        return "请上传基因映射文件", 400

    count_th = float(request.form.get("th_con") or 0.8)
    threshold_f = float(request.form.get("th_filter") or 0.1)
    threshold_a = float(request.form.get("th_add") or 0.9)
    f_max = float(request.form.get("f_max") or 0.7)
    f_min = float(request.form.get("f_min") or 0.0)
    a_max = float(request.form.get("a_max") or 1.0)
    a_min = float(request.form.get("a_min") or 0.8)

    model_name = request.form.get("model_name", "GCN")
    conv_channels = parse_int_list(request.form.get("conv_channels", "8,12,9"), default=[8, 12, 9])
    epoch = int(request.form.get("epoch") or 1)

    # Optional compatibility with your newer model.edge_pre(seed=...) function.
    # Existing template does not include this field, so default is a single seed 42.
    seeds = parse_seed_list(request.form.get("seeds", request.form.get("seed", "")), default=[42])

    job_dir = create_job_dir("edge")
    input_dir = ensure_dir(job_dir / "input")
    feature_dir = ensure_dir(job_dir / "expression")

    adj_path = save_uploaded_file(adj_file, input_dir)
    gene_map_path = save_uploaded_file(gene_map_file, input_dir)
    prepared_gene_dir = prepare_uploaded_data(gene_file, feature_dir, flatten_zip=False, unwrap_root=True)

    output_dirs: list[Path] = []
    for seed in seeds:
        try:
            edge_pre(
                str(adj_path),
                str(prepared_gene_dir),
                str(gene_map_path),
                model_name,
                list(conv_channels),
                threshold_f,
                threshold_a,
                f_max,
                f_min,
                a_max,
                a_min,
                epoch,
                str(GENE_LIST_PATH),
                seed=seed,
            )
        except TypeError:
            # Older PyGeoNet edge_pre has no seed argument.
            edge_pre(
                str(adj_path),
                str(prepared_gene_dir),
                str(gene_map_path),
                model_name,
                list(conv_channels),
                threshold_f,
                threshold_a,
                f_max,
                f_min,
                a_max,
                a_min,
                epoch,
                str(GENE_LIST_PATH),
            )
            break

    prefix = f"result_data_{model_name}_{threshold_f}_{threshold_a}"
    output_dirs = [p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not output_dirs:
        return "边预测已执行，但未找到输出结果目录，请检查模型函数输出路径。", 500

    package_dir = ensure_dir(job_dir / "edge_result")
    for result_dir in output_dirs:
        target = package_dir / result_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(result_dir, target)
        write_run_config(
            target / "run_config.txt",
            {
                "count_threshold": count_th,
                "model_name": model_name,
                "conv_channels": conv_channels,
                "epoch": epoch,
                "threshold_filter": threshold_f,
                "threshold_add": threshold_a,
                "filter_max": f_max,
                "filter_min": f_min,
                "add_max": a_max,
                "add_min": a_min,
                "seed_or_seeds": seeds,
                "gene_list": str(GENE_LIST_PATH),
            },
        )

    zip_path = make_zip(package_dir, "edge_result")
    return send_zip(zip_path)


@app.route("/health")
def health():
    prefer_project_root()
    purge_wrong_utils_package()
    utils_mod = import_project_module("utils")
    return jsonify(
        {
            "status": "ok",
            "app_dir": str(BASE_DIR),
            "project_root": str(PROJECT_ROOT),
            "runtime_dir": str(RUNTIME_DIR),
            "gene_list_path": str(GENE_LIST_PATH),
            "utils_imported_from": getattr(utils_mod, "__file__", None),
            "sys_path_first_items": sys.path[:5],
        }
    )


if __name__ == "__main__":
    host = os.environ.get("PYGEONET_HOST", "127.0.0.1")
    port = int(os.environ.get("PYGEONET_PORT", "5000"))
    debug = os.environ.get("PYGEONET_DEBUG", "1") not in {"0", "false", "False"}
    app.run(host=host, port=port, debug=debug)
