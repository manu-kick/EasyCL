
class Config():
    def __init__(self) -> None:
        
        self.batch_size = 10
        self.num_iterations = 10000
        self.device='cuda'
        self.seed = 42
        self.loss_type =   'anchor_align_unif'  #'centroids'   #anchor or volume or centroids or area 
        self.anchor_selection = 'text' #'visual' 'audio'  If loss_type = anchor you can choose the anchor type 
        
        self.lr = 1e-4
        self.eval_type=    'centroids'    #centroids or volume or area
        self.normalization =  True
        self.distribution_type=    'ce'           #ce or kl or wd
        self.similarity_matrix =     'everywhere'           #false or only_centroids or everywhere
        self.similarity_type =  'ordered_numbers_circle' #'2clusters' # 'normal' 'ordered_numbers ordered_numbers_circle
        self.softmax_similarity = True #False
        self.similarity_temperature = 0.5

        self.detach_centroids = True
        self.normalization_centroids = True
        self.diag_centroid_normalized = False
        self.centroids_matrix_temperature = True
        self.label_smoothing_centroids = False
        self.centroid_scale = 1#0.1

        self.wandb = True
        self.run =  f'RUN_={self.detach_centroids}_{self.loss_type}_{self.distribution_type}_{self.similarity_matrix}_{self.similarity_type}_{self.centroid_scale}{self.lr}'


        self.contra_temp_init = 0.1 #0.07
        self.contra_temp_learnable = False


        #AUDIO
        self.sample_rate=48000
        self.n_fft=512
        self.n_mels=64



        #PLOTS
        self.plot_path = './plots_ring'


    def log_config(self):
        # Convert class attributes to a dictionary
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        return config_dict
