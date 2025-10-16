class Config:
    # Base directory for all project files (adjust as needed)
    BASE_DIR = "/media/hdd/usr/leyla/Unlearn-Saliency"

    # Analysis and target paths for 10% forgetting
    ANALYSIS_10PCT = f"{BASE_DIR}/analysis/results/multi_experiment_analysis/10percent/real_influence_summary.json"
    TARGETS_10PCT = f"{BASE_DIR}/analysis/results/multi_experiment_analysis/10percent/lucent_targets_real.json"

    # Model paths
    UNLEARNED_MODEL = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    PRETRAINED_MODEL = f"{BASE_DIR}/experiments/models/resnet50_pretrained.pth"
    EVAL_UNLEARNED_MODEL = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLeval_result.pth.tar"


