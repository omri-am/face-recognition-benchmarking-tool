from FacesBenchmarkUtils import *
from clipModel import *
from dinoModel import *

def main():
    torch.cuda.empty_cache()
    ## models ##
    
    clip_model = CLIPModel("clip")
    vit8 = DinoModel("dino-vitb8")
    vit16 = DinoModel("dino-vitb16", version = "facebook/dino-vitb16")
    dinoV2 = DinoModel("dinov2-large", version = "facebook/dinov2-large")

    ## tasks ##

    upright_path = os.getcwd() + '/tests_datasets/inversion/lfw_test_pairs_300_upright_same.csv'
    inverted_path = os.getcwd() + '/tests_datasets/inversion/lfw_test_pairs_300_inverted_same.csv'
    inversion_images_path = os.getcwd() + '/tests_datasets/inversion/stimuliLabMtcnn/'

    sim_pairs_file_path = os.getcwd() + '/tests_datasets/similarity_perception/faces_visual_perception_similarity_behavioral_summary.csv'
    sim_images_path = os.getcwd() + '/tests_datasets/similarity_perception/intFacesLabMtcnn'

    upright_acc = AccuracyTask(
        pairs_file_path = upright_path,
        images_path = inversion_images_path,
        true_label = "same",
        distance_metric = pairwise.cosine_distances,
        name = "Upright_Accuracy")
    inverted_acc = AccuracyTask(
        pairs_file_path = inverted_path,
        images_path = inversion_images_path,
        true_label = "same",
        distance_metric = pairwise.cosine_distances,
        name = "Inverted_Accuracy")
    same_diff_task = CorrelationTask(
        pairs_file_path = sim_pairs_file_path,
        images_path = sim_images_path,
        name = "Same_Diff",
        distance_metric = pairwise.cosine_distances,
        correlation_metric = np.corrcoef)

    ## multi model manager ##

    multimodel_manager = MultiModelTaskManager(
        # models = [dino_model], 
        models = [vit16, vit8, dinoV2],
        tasks=[upright_acc, inverted_acc, same_diff_task])

    multimodel_manager.run_all_tasks_all_models()

if __name__ == "__main__":
    main()