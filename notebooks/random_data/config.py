class Config:
    # Base directory for all project files (adjust as needed)
    BASE_DIR = "/media/hdd/usr/leyla/Unlearn-Saliency"

    # Paths for pretrained model
    EVAL_PRETRAINED_MODEL = f"{BASE_DIR}/experiments/results/good_results/pretrained_resnet50/RLeval_result.pth.tar"
    PRETRAINED_MODEL = f"{BASE_DIR}/experiments/models/resnet50_pretrained.pth"

    # Analysis and target paths for 10% forgetting
    ANALYSIS_10PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/10percent/influence_summary.json"
    TARGETS_10PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/10percent/lucent_targets_10percent.json"
    CHANNELS_10PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/10percent/channels.csv"
    LAYERS_10PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/10percent/layers.csv"

    # Analysis and target paths for 20% forgetting
    ANALYSIS_20PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/20percent/influence_summary.json"
    TARGETS_20PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/20percent/lucent_targets_20percent.json"
    CHANNELS_20PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/20percent/channels.csv"
    LAYERS_20PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/20percent/layers.csv"

    # Analysis and target paths for 30% forgetting
    ANALYSIS_30PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/30percent/influence_summary.json"
    TARGETS_30PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/30percent/lucent_targets_30percent.json"
    CHANNELS_30PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/30percent/channels.csv"
    LAYERS_30PCT = f"{BASE_DIR}/analysis/results/unlearning_analysis/30percent/layers.csv"


    # Paths for unlearned models and their evaluations
    UNLEARNED_MODEL_10PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_10PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/RLeval_result.pth.tar"

    UNLEARNED_MODEL_20PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_20PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/RLeval_result.pth.tar"

    UNLEARNED_MODEL_30PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLcheckpoint.pth.tar"
    EVAL_UNLEARNED_MODEL_30PCT = f"{BASE_DIR}/experiments/results/good_results/random_forgetting_30percent_RL_tweak_conservative/RLeval_result.pth.tar"


    # Experiment comparison path
    EXPERIMENT_COMPARISON_PATH = f"{BASE_DIR}/analysis/results/unlearning_analysis/experiment_comparison.json"



    #CLASS-WISE RESULTS PATHS
    
    #CATS:
    CATS_04_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_4_salun/RLeval_result.pth.tar"
    CATS_05_CONSERVATIVE_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_5_conservative/RLeval_result.pth.tar"
    CATS_05_FT_CONSERVATIVE_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_5_finetune_conservative/FTeval_result.pth.tar"
    CATS_05_GA_EXTREME_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_5_GA_extreme/GAeval_result.pth.tar"
    CATS_05_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_5_salun/RLeval_result.pth.tar"
    CATS_05_ULTRA_CONSERVATIVE_PATH = f"{BASE_DIR}/experiments/models/resnet50_cats_forgetting/mask0_5_ultra_conservative/RLeval_result.pth.tar"

    #DOGS:
    DOGS_04_PATH = f"{BASE_DIR}/experiments/models/resnet50_dogs_forgetting/mask0_4/RLeval_result.pth.tar"
    DOGS_05_CONSERVATIVE_PATH = f"{BASE_DIR}/experiments/models/resnet50_dogs_forgetting/mask0_5_GA_method/GAeval_result.pth.tar"
    DOGS_05_GA_PATH = f"{BASE_DIR}/experiments/models/resnet50_dogs_forgetting/mask0_5_salun/GAeval_result.pth.tar"
    DOGS_05_RL_PATH = f"{BASE_DIR}/experiments/models/resnet50_dogs_forgetting/mask0_5_salun/RLeval_result.pth.tar"
    DOGS_06_PATH = f"{BASE_DIR}/experiments/models/resnet50_dogs_forgetting/mask0_6/RLeval_result.pth.tar"

    #VEHICLES:
    VEHICLES_05_CONSERVATIVE_PATH = f"{BASE_DIR}/experiments/models/resnet50_vehicles_forgetting/mask0_5_conservative/RLeval_result.pth.tar"
    VEHICLES_05_GA_PATH = f"{BASE_DIR}/experiments/models/resnet50_vehicles_forgetting/mask0_5_GA_method/GAeval_result.pth.tar"


    #INCEPTIONV3 CATS:
    INCEPTIONV3_CATS_05_PATH = f"{BASE_DIR}/class_wise_forgetting/result/RLeval_inception05_cats.pth.tar"
    INCEPTIONV3_CATS_03_PATH = f"{BASE_DIR}/class_wise_forgetting/result/RLeval_inception03_cats.pth.tar"

    