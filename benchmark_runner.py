from FacesBenchmarkUtils import *
from clipModel import *
from dinoModel import *
from vgg16Model import *

def main():
    torch.cuda.empty_cache()
    ## models ##
    
    clip_model = CLIPModel('clip')
    vit8 = DinoModel('dino-vitb8', 'facebook/dino-vitb8')
    vit16 = DinoModel('dino-vitb16', version = 'facebook/dino-vitb16')
    dinoV2 = DinoModel('dinov2-base', version = 'facebook/dinov2-base')
    vgg16_trained_fc7 = Vgg16Model(name='VGG16-trained-fc7',
                                   weights_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth',
                                   extract_layer='classifier.5')
    vgg16_trained_avgpool = Vgg16Model(name='VGG16-trained-avgpool',
                                   weights_path='/home/new_storage/experiments/face_memory_task/models/face_trained_vgg16_119.pth', 
                                   extract_layer='avgpool')
    vgg16_untrained_fc7 = Vgg16Model(name='VGG16-untrained-fc7',
                                     extract_layer='classifier.5')
    vgg16_untrained_avgpool = Vgg16Model(name='VGG16-untrained-avgpool',
                                         extract_layer='avgpool')

    ## tasks ##

    lfw_pairs = os.path.join(os.getcwd(), 'tests_datasets/LFW/lfw_test_pairs_only_img_names.txt')
    lfw_images = os.path.join(os.getcwd(), 'tests_datasets/LFW/lfw-align-128')

    lfw_acc = AccuracyTask(
        pairs_file_path=lfw_pairs,
        images_path=lfw_images,
        true_label='same',
        distance_metric=pairwise.cosine_distances,
        name='LFW'
    )

    upright_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv')
    inverted_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv')
    inversion_images_path = os.path.join(os.getcwd(), 'tests_datasets/inversion/stimuliLabMtcnn/')

    upright_acc = AccuracyTask(
        pairs_file_path = upright_path,
        images_path = inversion_images_path,
        true_label = 'same',
        distance_metric = pairwise.cosine_distances,
        name = 'Upright Accuracy'
        )
    inverted_acc = AccuracyTask(
        pairs_file_path = inverted_path,
        images_path = inversion_images_path,
        true_label = 'same',
        distance_metric = pairwise.cosine_distances,
        name = 'Inverted Accuracy'
        )
    
    sim_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception/faces_visual_perception_similarity_behavioral_summary.csv')
    sim_DP_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception/faces_visual_perception_similarity_behavioral_summary_DP.csv')
    sim_SP_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception/faces_visual_perception_similarity_behavioral_summary_SP.csv')
    sim_images_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception/intFacesLabMtcnn')
    
    same_diff_task = CorrelationTask(
        pairs_file_path = sim_pairs_file_path,
        images_path = sim_images_path,
        name = 'Same Diff',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    same_diff_DP_task = CorrelationTask(
        pairs_file_path = sim_DP_pairs_file_path,
        images_path = sim_images_path,
        name = 'Same Diff DP',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    same_diff_SP_task = CorrelationTask(
        pairs_file_path = sim_SP_pairs_file_path,
        images_path = sim_images_path,
        name = 'Same Diff SP',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    
    sim_international_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary.csv')
    sim_international_DP_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_DP.csv')
    sim_international_SP_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/faces_visual_perception_similarity_behavioral_summary_SP.csv')
    sim_international_images_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_international_celebs/intFacesLabMtcnn')
    
    same_diff_int_task = CorrelationTask(
        pairs_file_path = sim_international_pairs_file_path,
        images_path = sim_international_images_path,
        name = 'Same Diff: International Celebs',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    same_diff_DP_int_task = CorrelationTask(
        pairs_file_path = sim_international_DP_pairs_file_path,
        images_path = sim_international_images_path,
        name = 'Same Diff DP: International Celebs',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    same_diff_SP_int_task = CorrelationTask(
        pairs_file_path = sim_international_SP_pairs_file_path,
        images_path = sim_international_images_path,
        name = 'Same Diff SP: International Celebs',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef
        )
    
    sim_il_familiar_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_familiar_distances.csv')
    sim_il_unfamiliar_pairs_file_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/israeli_new_images_perception_unfamiliar_distances.csv')
    sim_il_images_path = os.path.join(os.getcwd(), 'tests_datasets/similarity_perception_israeli_celebs/newIsraeliFacesStimuliLabMtcnn')

    familiar_il_task = CorrelationTask(
        pairs_file_path = sim_il_familiar_pairs_file_path,
        images_path = sim_il_images_path,
        name = 'IL Celebs: Familiar Performance',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef 
    )
    unfamiliar_il_task = CorrelationTask(
        pairs_file_path = sim_il_unfamiliar_pairs_file_path,
        images_path = sim_il_images_path,
        name = 'IL Celebs: Unfamiliar Performance',
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef 
    )

    caucasian_pairs_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/vggface_other_race_same_caucasian.csv')
    asian_pairs_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/vggface_other_race_same_asian.csv')
    other_race_images_path = os.path.join(os.getcwd(), 'tests_datasets/other_race/other_raceLabMtcnn')

    other_race_caucasian = AccuracyTask(
        pairs_file_path = caucasian_pairs_path,
        images_path = other_race_images_path,
        true_label = 'same',
        distance_metric = pairwise.cosine_distances,
        name = 'Caucasian Accuracy'
        )
    other_race_asian = AccuracyTask(
        pairs_file_path = asian_pairs_path,
        images_path = other_race_images_path,
        true_label = 'same',
        distance_metric = pairwise.cosine_distances,
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
        distance_metric = pairwise.cosine_distances,
        name = 'Thatcher Effect'
    )

    ## multi model manager ##

    multimodel_manager = MultiModelTaskManager(
        models = [vgg16_trained_fc7, 
                  vgg16_trained_avgpool, 
                  vgg16_untrained_fc7, 
                  vgg16_untrained_avgpool,
                  clip_model, 
                  dinoV2, 
                  vit8, 
                  vit16],
        tasks = [lfw_acc,
                 upright_acc, 
                 inverted_acc, 
                 same_diff_task, 
                 same_diff_DP_task,
                 same_diff_SP_task,
                 same_diff_int_task,
                 same_diff_DP_int_task,
                 same_diff_SP_int_task,
                 familiar_il_task,
                 unfamiliar_il_task,
                 other_race_caucasian, 
                 other_race_asian,
                 thatcher_task])

    multimodel_manager.run_all_tasks_all_models(export_path = os.path.join(os.getcwd(), f'results/{date.today()}'))

if __name__ == '__main__':
    main()