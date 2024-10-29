from facesBenchmarkUtils import *
from models import *
from tasks import *
from datetime import datetime, date

def batch_cosine_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    cosine_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=1)
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def main():
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
                        'classifier.6']
    
    vgg16_trained = Vgg16Model(name='VGG16-trained',
                               weights_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth',
                               extract_layers=['avgpool', 'classifier.5'])
    vgg16_trained_all = Vgg16Model(name='VGG16-trained',
                               weights_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth',
                               extract_layers=all_vgg16_layers)
    vgg16_untrained = Vgg16Model(name='VGG16-untrained',
                                 extract_layers=['avgpool', 'classifier.5'])
    vgg16_untrained_all = Vgg16Model(name='VGG16-untrained',
                                 extract_layers=all_vgg16_layers)
    clip_model = CLIPModel('clip')
    # vit8 = DinoModel('dino-vitb8', 'facebook/dino-vitb8')
    # vit16 = DinoModel('dino-vitb16', version = 'facebook/dino-vitb16')
    dinoV2 = DinoModel('dinov2-base', version = 'facebook/dinov2-base')

    ## tasks ##

    lfw_pairs = os.path.join(os.getcwd(), 'tests_datasets/LFW/lfw_test_pairs_only_img_names.txt')
    lfw_images = os.path.join(os.getcwd(), 'tests_datasets/LFW/lfw-align-128')

    lfw_acc = AccuracyTask(
        pairs_file_path=lfw_pairs,
        images_path=lfw_images,
        true_label='same',
        distance_metric=batch_cosine_distance,
        name='LFW'
    )

    upright_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv')
    inverted_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv')
    inversion_images_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/stimuliLabMtcnn/')

    upright_acc = AccuracyTask(
        pairs_file_path = upright_path,
        images_path = inversion_images_path,
        true_label = 'same',
        distance_metric = batch_cosine_distance,
        name = 'Upright'
        )
    inverted_acc = AccuracyTask(
        pairs_file_path = inverted_path,
        images_path = inversion_images_path,
        true_label = 'same',
        distance_metric = batch_cosine_distance,
        name = 'Inverted'
        )
    
    sim_international_memory_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_memory_visual_similarity_behavioral_summary.csv')
    sim_international_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv')
    sim_international_DP_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_DP.csv')
    sim_international_SP_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_SP.csv')
    sim_international_images_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn')
    
    same_diff_visual_int_task = CorrelationTask(
        pairs_file_path = sim_international_pairs,
        images_path = sim_international_images_path,
        name = 'Visual Perception Similarity: International Celebs',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )
    same_diff_memory_int_task = CorrelationTask(
        pairs_file_path = sim_international_memory_pairs,
        images_path = sim_international_images_path,
        name = 'Memory Perception Similarity: International Celebs',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )
    same_diff_DP_int_task = CorrelationTask(
        pairs_file_path = sim_international_DP_pairs,
        images_path = sim_international_images_path,
        name = 'Visual Perception Similarity DP: International Celebs',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )
    same_diff_SP_int_task = CorrelationTask(
        pairs_file_path = sim_international_SP_pairs,
        images_path = sim_international_images_path,
        name = 'Visual Perception Similarity SP: International Celebs',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )
    
    sim_il_familiar_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv')
    sim_il_unfamiliar_pairs = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv')
    sim_il_images_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn')

    familiar_il_task = CorrelationTask(
        pairs_file_path = sim_il_familiar_pairs,
        images_path = sim_il_images_path,
        name = 'IL Celebs: Familiar Performance',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef 
    )
    unfamiliar_il_task = CorrelationTask(
        pairs_file_path = sim_il_unfamiliar_pairs,
        images_path = sim_il_images_path,
        name = 'IL Celebs: Unfamiliar Performance',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef 
    )

    caucasian_pairs_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/vggface_other_race_same_caucasian.csv')
    asian_pairs_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/vggface_other_race_same_asian.csv')
    other_race_images_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/other_raceLabMtcnn')

    other_race_caucasian = AccuracyTask(
        pairs_file_path = caucasian_pairs_path,
        images_path = other_race_images_path,
        true_label = 'same',
        distance_metric = batch_cosine_distance,
        name = 'Caucasian Accuracy'
        )
    other_race_asian = AccuracyTask(
        pairs_file_path = asian_pairs_path,
        images_path = other_race_images_path,
        true_label = 'same',
        distance_metric = batch_cosine_distance,
        name = 'Asian Accuracy'
        )
    
    thatcher_inverted_pairs = os.path.join(os.getcwd(), 'tests_datasets/thatcher/human_ratings_thatcher_inverted.csv')
    thatcher_upright_pairs = os.path.join(os.getcwd(), 'tests_datasets/thatcher/human_ratings_thatcher_upright.csv')
    thatcher_combined_pairs = os.path.join(os.getcwd(), 'tests_datasets/thatcher/human_ratings_thatcher_combined.csv')
    thatcher_images_path = os.path.join(os.getcwd(), 'tests_datasets/thatcher/images_thatcher_mtcnn')

    thatcher_task = RelativeDifferenceTask(
        pairs_file_path = thatcher_combined_pairs,
        images_path = thatcher_images_path,
        group_column = 'cond',
        distance_metric = torch.cdist,
        name = 'Thatcher Effect'
        )
    
    conditioned_pairs = os.path.join(os.getcwd(), 'tests_datasets/critical_features/critical_features_all_conditions.csv')
    critical_distances_pairs = os.path.join(os.getcwd(), 'tests_datasets/critical_features/critical_features_critical_distances.csv')
    noncritical_distances_pairs = os.path.join(os.getcwd(), 'tests_datasets/critical_features/critical_features_noncritical_distances.csv')
    conditioned_images_path = os.path.join(os.getcwd(), 'tests_datasets/critical_features/img_dataset/joined')

    critical_features_conditioned = ConditionedAverageDistances(
        pairs_file_path = conditioned_pairs,
        images_path = conditioned_images_path,
        condition_column = 'cond',
        distance_metric = batch_cosine_distance,
        normalize = False,
        name = 'Critical Features'
        )

    critical_features_conditioned_normalized = ConditionedAverageDistances(
        pairs_file_path = conditioned_pairs,
        images_path = conditioned_images_path,
        condition_column = 'cond',
        distance_metric = batch_cosine_distance,
        normalize = True,
        name = 'Critical Features Normalized'
        )
    
    critical_features_critical_dis = CorrelationTask(
        pairs_file_path = critical_distances_pairs,
        images_path = conditioned_images_path,
        name = 'Critical Features Critical Distances',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )
    
    critical_features_noncritical_dis = CorrelationTask(
        pairs_file_path = noncritical_distances_pairs,
        images_path = conditioned_images_path,
        name = 'Critical Features Non-Critical Distances',
        distance_metric = batch_cosine_distance,
        correlation_metric = np.corrcoef
        )

    ## multi model manager ##

    multimodel_manager = MultiModelTaskManager(
        models = [
            # vgg16_trained_all,
            vgg16_trained,
            # vgg16_untrained_all, 
            vgg16_untrained,
            clip_model, 
            dinoV2, 
            # vit8, 
            # vit16
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
            critical_features_conditioned,
            critical_features_conditioned_normalized,
            critical_features_critical_dis,
            critical_features_noncritical_dis
            ])
    
    multimodel_manager2 = MultiModelTaskManager(
        models = [
            vgg16_trained_all,
            # vgg16_trained,
            # vgg16_untrained_all, 
            # vgg16_untrained,
            # clip_model, 
            # dinoV2, 
            # vit8, 
            # vit16
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
            critical_features_conditioned,
            critical_features_conditioned_normalized,
            critical_features_critical_dis,
            critical_features_noncritical_dis
            ])

    path = os.path.join(os.getcwd(),'results',f'{date.today()}',f"{datetime.now().strftime('%H%M')}")
    multimodel_manager.run_all_tasks_all_models(export_path=path, print_log=True)
    path = os.path.join(os.getcwd(),'results',f'{date.today()}',f"{datetime.now().strftime('%H%M')}")
    multimodel_manager2.run_all_tasks_all_models(export_path=path, print_log=True)

if __name__ == '__main__':
    main()