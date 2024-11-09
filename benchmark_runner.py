from facesBenchmarkUtils import *
from models import *
from tasks import *
from datetime import datetime, date

torch.cuda.empty_cache()
## models ##

all_vgg16_layers = ['features.0',
                    'features.1',
                    'features.2',
                    'features.3',
                    'features.4',
                    'features.5',
                    'features.6',
                    'features.7',
                    'features.8',
                    'features.9',
                    'features.10',
                    'features.11',
                    'features.12',
                    'features.13',
                    'features.14',
                    'features.15',
                    'features.16',
                    'features.17',
                    'features.18',
                    'features.19',
                    'features.20',
                    'features.21',
                    'features.22',
                    'features.23',
                    'features.24',
                    'features.25',
                    'features.26',
                    'features.27',
                    'features.28',
                    'features.29',
                    'features.30',
                    'avgpool',
                    'classifier.0',
                    'classifier.1',
                    'classifier.2',
                    'classifier.3',
                    'classifier.4',
                    'classifier.5',
                    'classifier.6'
                    ]

vgg16_trained = Vgg16Model(model_name='VGG16-trained',
                            weights_file_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth',
                            layers_to_extract=['avgpool', 'classifier.5'])
vgg16_trained_all = Vgg16Model(model_name='VGG16-trained',
                            weights_file_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth',
                            layers_to_extract=all_vgg16_layers)
vgg16_untrained = Vgg16Model(model_name='VGG16-untrained',
                                layers_to_extract=['avgpool', 'classifier.5'])
vgg16_untrained_all = Vgg16Model(model_name='VGG16-untrained',
                                layers_to_extract=all_vgg16_layers)
clip_model = CLIPModel('clip')
dinoV2 = DinoModel('dinov2-base', version = 'facebook/dinov2-base')

## tasks ##

lfw_pairs = './tests_datasets/LFW/lfw_test_pairs_only_img_names.txt'
lfw_images = './tests_datasets/LFW/lfw-align-128'

lfw_acc = AccuracyTask(
    pairs_file_path=lfw_pairs,
    images_folder_path=lfw_images,
    true_label='same',
    distance_function=pairwise.cosine_distances,
    task_name = 'LFW'
)

upright_path = './tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv'
inverted_path = './tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv'
inversion_images_folder_path = './tests_datasets/inversion/stimuliLabMtcnn/'

upright_acc = AccuracyTask(
    pairs_file_path = upright_path,
    images_folder_path = inversion_images_folder_path,
    true_label = 'same',
    distance_function = pairwise.cosine_distances,
    task_name = 'Inversion Effect - Upright'
    )
inverted_acc = AccuracyTask(
    pairs_file_path = inverted_path,
    images_folder_path = inversion_images_folder_path,
    true_label = 'same',
    distance_function = pairwise.cosine_distances,
    task_name = 'Inversion Effect - Inverted'
    )

sim_international_memory_pairs = './tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv'
sim_international_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv'
sim_international_DP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_DP.csv'
sim_international_SP_pairs = './tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_SP.csv'
sim_international_images_folder_path = './tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn'

same_diff_visual_int_task = CorrelationTask(
    pairs_file_path = sim_international_pairs,
    images_folder_path = sim_international_images_folder_path,
    task_name = 'International Celebs - Visual Perception Similarity',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )
same_diff_memory_int_task = CorrelationTask(
    pairs_file_path = sim_international_memory_pairs,
    images_folder_path = sim_international_images_folder_path,
    task_name = 'International Celebs - Memory Perception Similarity',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )
same_diff_DP_int_task = CorrelationTask(
    pairs_file_path = sim_international_DP_pairs,
    images_folder_path = sim_international_images_folder_path,
    task_name = 'International Celebs - Visual Perception Similarity DP',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )
same_diff_SP_int_task = CorrelationTask(
    pairs_file_path = sim_international_SP_pairs,
    images_folder_path = sim_international_images_folder_path,
    task_name = 'International Celebs - Visual Perception Similarity SP',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )

sim_il_familiar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv'
sim_il_unfamiliar_pairs = './tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv'
sim_il_images_folder_path = './tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn'

familiar_il_task = CorrelationTask(
    pairs_file_path = sim_il_familiar_pairs,
    images_folder_path = sim_il_images_folder_path,
    task_name = 'IL Celebs - Familiar Performance',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef 
)
unfamiliar_il_task = CorrelationTask(
    pairs_file_path = sim_il_unfamiliar_pairs,
    images_folder_path = sim_il_images_folder_path,
    task_name = 'IL Celebs - Unfamiliar Performance',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef 
)

caucasian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_caucasian.csv'
asian_pairs_path = './tests_datasets/other_race/vggface_other_race_same_asian.csv'
other_race_images_folder_path = './tests_datasets/other_race/other_raceLabMtcnn'

other_race_caucasian = AccuracyTask(
    pairs_file_path = caucasian_pairs_path,
    images_folder_path = other_race_images_folder_path,
    true_label = 'same',
    distance_function = pairwise.cosine_distances,
    task_name = 'Other Race Effect - Caucasian'
    )
other_race_asian = AccuracyTask(
    pairs_file_path = asian_pairs_path,
    images_folder_path = other_race_images_folder_path,
    true_label = 'same',
    distance_function = pairwise.cosine_distances,
    task_name = 'Other Race Effect - Asian'
    )

thatcher_combined_pairs = './tests_datasets/thatcher/human_ratings_thatcher_combined.csv'
thatcher_images_folder_path = './tests_datasets/thatcher/images_thatcher_mtcnn'

thatcher_task = RelativeDifferenceTask(
    pairs_file_path = thatcher_combined_pairs,
    images_folder_path = thatcher_images_folder_path,
    group_column = 'cond',
    distance_function = torch.cdist,
    task_name = 'Thatcher Effect'
    )

conditioned_pairs = './tests_datasets/critical_features/critical_features_all_conditions.csv'
critical_distances_pairs = './tests_datasets/critical_features/critical_features_critical_distances.csv'
noncritical_distances_pairs = './tests_datasets/critical_features/critical_features_noncritical_distances.csv'
conditioned_images_folder_path = './tests_datasets/critical_features/img_dataset/joined'

critical_features_conditioned_normalized = ConditionedAverageDistances(
    pairs_file_path = conditioned_pairs,
    images_folder_path = conditioned_images_folder_path,
    condition_column = 'cond',
    distance_function = pairwise.cosine_distances,
    normalize = True,
    task_name = 'Critical Features'
    )

critical_features_critical_dis = CorrelationTask(
    pairs_file_path = critical_distances_pairs,
    images_folder_path = conditioned_images_folder_path,
    task_name = 'Critical Features - Critical Distances',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )

critical_features_noncritical_dis = CorrelationTask(
    pairs_file_path = noncritical_distances_pairs,
    images_folder_path = conditioned_images_folder_path,
    task_name = 'Critical Features - Non-Critical Distances',
    distance_function = pairwise.cosine_distances,
    correlation_function = np.corrcoef
    )

## multi model manager ##

multimodel_manager = MultiModelTaskManager(
    models = [
        vgg16_trained_all,
        vgg16_trained,
        vgg16_untrained_all, 
        vgg16_untrained,
        clip_model, 
        dinoV2, 
        ],
    tasks = [
        lfw_acc,
        upright_acc, 
        inverted_acc, 
        same_diff_visual_int_task,
        same_diff_memory_int_task,
        same_diff_DP_int_task,
        same_diff_SP_int_task,
        familiar_il_task,
        unfamiliar_il_task,
        other_race_caucasian, 
        other_race_asian,
        thatcher_task,
        critical_features_conditioned_normalized,
        critical_features_critical_dis,
        critical_features_noncritical_dis
        ])

path = os.path.join(os.getcwd(),'results',f'{date.today()}',f"{datetime.now().strftime('%H%M')}")
multimodel_manager.run_all_tasks_all_models(export_path=path, print_log=True)