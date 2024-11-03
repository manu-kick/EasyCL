
class Config():
    def __init__(self) -> None:
        
        self.batch_size = 10
        self.num_iterations = 10000
        self.device='cuda'
        self.seed = 42
        self.loss_type =   'centroids'  #'centroids'   #anchor or volume or centroids or area 
        self.anchor_selection = 'text' #'visual' 'audio'  If loss_type = anchor you can choose the anchor type 
        
        self.lr = 1e-4
        self.eval_type=    'centroids'    #centroids or volume or area
        self.normalization =  True
        self.distribution_type=    'ce'           #ce or kl or wass
        self.similarity_matrix =     'everywhere'           #false or only_centroids or everywhere
        self.similarity_type = 'normal' #'2clusters' # 'normal' 'ordered_numbers
        
        self.detach_centroids = True
        self.normalization_centroids = False
        self.softmax_similarity = False
        self.diag_centroid_normalized = False
        self.centroids_matrix_temperature = False
        self.label_smoothing_centroids = False
        self.centroid_scale = 4

        self.wandb = True
        self.run =  f'RUN_detach_{self.loss_type}_{self.distribution_type}_{self.similarity_matrix}_{self.similarity_type}'


        self.contra_temp_init = 0.07


        #AUDIO
        self.sample_rate=48000
        self.n_fft=512
        self.n_mels=64

    def log_config(self):
        # Convert class attributes to a dictionary
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        return config_dict