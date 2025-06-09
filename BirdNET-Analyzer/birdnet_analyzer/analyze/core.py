import os
from typing import List, Literal


def analyze(
    input: str,
    output: str | None = None,
    *,
    min_conf: float = 0.25,
    classifier: str | None = None,
    lat: float = -1,
    lon: float = -1,
    week: int = -1,
    slist: str | None = None,
    sensitivity: float = 1.0,
    overlap: float = 0,
    fmin: int = 0,
    fmax: int = 15000,
    audio_speed: float = 1.0,
    batch_size: int = 1,
    combine_results: bool = False,
    rtype: Literal["table", "audacity", "kaleidoscope", "csv"] = "csv",
    skip_existing_results: bool = False,
    sf_thresh: float = 0.03,
    top_n: int | None = None,
    merge_consecutive: int = 1,
    threads: int = 1,
    locale: str = "en",
):
    """
    分析音频文件并返回置信度最高的鸟类检测结果。
    
    Args:
        input (str): 输入音频文件路径
        min_conf (float): 最小置信度阈值，默认0.25
        ... (其他参数保持不变)
        
    Returns:
        dict: 包含以下字段的字典：
            - status: 'success' 或 'error'
            - species: 检测到的鸟类种类列表，每个种类包含：
                - name: 物种名称
                - confidence: 置信度
            - error: 如果status为error，则包含错误信息
    """
    from multiprocessing import Pool
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import analyze_file
    from birdnet_analyzer.utils import ensure_model_exists

    ensure_model_exists()

    # 设置参数
    flist = _set_params(
        input=input,
        output=None,
        min_conf=min_conf,
        custom_classifier=classifier,
        lat=lat,
        lon=lon,
        week=week,
        slist=slist,
        sensitivity=sensitivity,
        locale=locale,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        bs=batch_size,
        combine_results=False,
        rtype=rtype,
        sf_thresh=sf_thresh,
        top_n=top_n,
        merge_consecutive=merge_consecutive,
        skip_existing_results=skip_existing_results,
        threads=threads,
        labels_file=cfg.LABELS_FILE,
    )

    print(f"Found {len(cfg.FILE_LIST)} files to analyze")
    print(f"Species list contains {len(cfg.SPECIES_LIST) if cfg.SPECIES_LIST else len(cfg.LABELS)} species")

    all_detections = []
    processing_times = []

    # 分析文件
    if cfg.CPU_THREADS < 2 or len(flist) < 2:
        for entry in flist:
            result = analyze_file(entry)
            if result['status'] == 'success':
                all_detections.extend(result['detections'])
                processing_times.append(result['processing_time'])
            else:
                return result
    else:
        with Pool(cfg.CPU_THREADS) as p:
            results = p.map_async(analyze_file, flist)
            results.wait()
            for result in results.get():
                if result['status'] == 'success':
                    all_detections.extend(result['detections'])
                    processing_times.append(result['processing_time'])
                else:
                    return result

    # 提取唯一的鸟类种类，并保留最高置信度
    species_dict = {}
    for detection in all_detections:
        species = detection['species']
        confidence = detection['confidence']
        if species not in species_dict or confidence > species_dict[species]['confidence']:
            species_dict[species] = {
                'name': species,
                'confidence': confidence
            }

    # 转换为列表并按置信度排序
    species_list = sorted(
        species_dict.values(),
        key=lambda x: x['confidence'],
        reverse=True
    )

    # 只返回置信度最高的一个结果
    if species_list and species_list[0]['confidence'] >= min_conf:
        return {
            'status': 'success',
            'species': species_list[0]  # 只返回最高置信度的结果
        }
    else:
        return {
            'status': 'success',
            'species': None  # 如果没有检测到任何鸟类
        }


def _set_params(
    input,
    output,
    min_conf,
    custom_classifier,
    lat,
    lon,
    week,
    slist,
    sensitivity,
    locale,
    overlap,
    fmin,
    fmax,
    audio_speed,
    bs,
    combine_results,
    rtype,
    skip_existing_results,
    sf_thresh,
    top_n,
    merge_consecutive,
    threads,
    labels_file=None,
):
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.analyze.utils import load_codes
    from birdnet_analyzer.species.utils import get_species_list
    from birdnet_analyzer.utils import collect_audio_files, read_lines

    if not output:
        cfg.OUTPUT_PATH = "/tmp"
    else:
        cfg.OUTPUT_PATH = "/tmp"

    cfg.CODES = load_codes()
    cfg.LABELS = read_lines(labels_file if labels_file else cfg.LABELS_FILE)
    cfg.SKIP_EXISTING_RESULTS = skip_existing_results
    cfg.LOCATION_FILTER_THRESHOLD = sf_thresh
    cfg.TOP_N = top_n
    cfg.MERGE_CONSECUTIVE = merge_consecutive
    cfg.INPUT_PATH = input
    cfg.MIN_CONFIDENCE = min_conf
    cfg.SIGMOID_SENSITIVITY = sensitivity
    cfg.SIG_OVERLAP = overlap
    cfg.BANDPASS_FMIN = fmin
    cfg.BANDPASS_FMAX = fmax
    cfg.AUDIO_SPEED = audio_speed
    cfg.RESULT_TYPES = rtype
    cfg.COMBINE_RESULTS = combine_results
    cfg.BATCH_SIZE = bs

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = threads
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = threads

    if custom_classifier is not None:
        cfg.CUSTOM_CLASSIFIER = custom_classifier  # we treat this as absolute path, so no need to join with dirname

        if custom_classifier.endswith(".tflite"):
            cfg.LABELS_FILE = custom_classifier.replace(".tflite", "_Labels.txt")  # same for labels file

            if not os.path.isfile(cfg.LABELS_FILE):
                cfg.LABELS_FILE = custom_classifier.replace("Model_FP32.tflite", "Labels.txt")

            cfg.LABELS = read_lines(cfg.LABELS_FILE)
        else:
            cfg.APPLY_SIGMOID = False
            # our output format
            cfg.LABELS_FILE = os.path.join(custom_classifier, "labels", "label_names.csv")

            if not os.path.isfile(cfg.LABELS_FILE):
                cfg.LABELS_FILE = os.path.join(custom_classifier, "assets", "label.csv")
                cfg.LABELS = read_lines(cfg.LABELS_FILE)
            else:
                cfg.LABELS = [line.split(",")[1] for line in read_lines(cfg.LABELS_FILE)]
    else:
        cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = lat, lon, week
        cfg.CUSTOM_CLASSIFIER = None

        if cfg.LATITUDE == -1 and cfg.LONGITUDE == -1:
            if not slist:
                cfg.SPECIES_LIST_FILE = None
            else:
                cfg.SPECIES_LIST_FILE = slist

                if os.path.isdir(cfg.SPECIES_LIST_FILE):
                    cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, "species_list.txt")

            cfg.SPECIES_LIST = read_lines(cfg.SPECIES_LIST_FILE)
        else:
            cfg.SPECIES_LIST_FILE = None
            cfg.SPECIES_LIST = get_species_list(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)

    lfile = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(locale))
    )

    if locale not in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = read_lines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    return [(f, cfg.get_config()) for f in cfg.FILE_LIST]
