import json

class Config():
    def __init__(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        
        """ hypermeter for dataloader """
        self.datapath = config["datapath"]
        self.resize_size = config["resize_size"]
        self.crop_size = config["crop_size"]
        self.channel = config["channel"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.buffer_size = config["buffer_size"]
        self.repeat = config["repeat"]


        """ hypermerter for training """
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.beta1 = config["beta1"]
        self.save_epoch_freq = config["save_epoch_freq"]
        
        self.cycle_weight = config["cycle_weight"]
        self.idenity_weight = config["idenity_weight"]
        self.loss_mode = config["loss_mode"]

        """ hypermeter for networks"""
        self.input_nc = config["input_nc"]
        self.output_nc = config["output_nc"]
        self.ngf = config["ngf"]
        self.ndf = config["ndf"]
        self.n_blocks = config["n_blocks"]
        self.n_downsampling = config["n_downsampling"]
        self.n_layers = config["n_layers"]
        self.norm = config["norm"]
        self.padding = config["padding"]

        self.log_path = config["log_path"]
        self.is_Distributed = config["is_Distributed"]

        