class Config:
    # Base directory for all project files (adjust as needed)
    BASE_DIR = "/media/hdd/usr/leyla/Unlearn-Saliency"

    # Paths for pretrained model
    EVAL_PRETRAINED_MODEL = f"{BASE_DIR}/experiments/results/good_results/pretrained_resnet50/RLeval_result.pth.tar"
    PRETRAINED_MODEL = f"{BASE_DIR}/experiments/models/resnet50_pretrained.pth"

    # Analysis and target paths for 10% forgetting
    ANALYSIS_10PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/10percent/influence_summary.json"
    TARGETS_10PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/10percent/lucent_targets_10percent.json"
    CHANNELS_10PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/10percent/channels.csv"
    LAYERS_10PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/10percent/layers.csv"

    # Analysis and target paths for 20% forgetting
    ANALYSIS_20PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/20percent/influence_summary.json"
    TARGETS_20PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/20percent/lucent_targets_20percent.json"

    # Analysis and target paths for 30% forgetting
    ANALYSIS_30PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/30percent/influence_summary.json"
    TARGETS_30PCT = f"{BASE_DIR}/analysis/results/deterministic_analysis/30percent/lucent_targets_30percent.json"
    

    # Paths for unlearned models and their evaluations
    UNLEARNED_MODEL_10PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_10PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLeval_result.pth.tar"

    UNLEARNED_MODEL_20PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_20PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLeval_result.pth.tar"

    UNLEARNED_MODEL_30PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_30PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLeval_result.pth.tar"


