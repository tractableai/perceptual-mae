import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as torchvision_models  # vision models lib for image encoders
import timm # vision transformers lib for image encoders
# hugging face lib for language encoders
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from omegaconf import OmegaConf

def build_image_encoder(arch='resnet18', pretrained=True):
    # get all the model names in torchvision library
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))

    # Get all the model names in pytorch image models (timm) library
    timm_model_names = sorted(timm.list_models(pretrained=True))

    # by default timm is used, however this can be changed should you choose to use torchvision;
    if arch in timm_model_names:
        # set num_classes to 0 to get the penuiltimate layer outputs; or physically change last layer in the encoder model;
        enc_model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        
    elif arch in torchvision_model_names:
        enc_model = torch.hub.load('pytorch/vision', arch, pretrained=pretrained)
    else:
        raise Exception('the following model architecture: {} has not been implemented'.format(arch))
    
    return enc_model 


def build_language_encoder(arch='roberta-base', task='generic', pretrained=True, lang_model_config=None):
    # initialise model with the config;
    # Note: Loading a model from its configuration file does not load the model weights. 
    # It only affects the model’s configuration. Use from_pretrained()to load the model weights.
    """
    e.g. 
    # The following Downloads model and configuration from huggingface.co and cache.
    model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
    https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMaskedLM
    """
    if pretrained==True:
        enc_config = AutoConfig.from_pretrained(arch)
        
        # GENERIC
        if task=='generic' or task==None:
            # This is a generic model class that will be instantiated as one of the base model classes of the library when created with the
            if lang_model_config!=None:
                enc_config = update_huggingface_config(lang_model_config, enc_config)
                # updating the pretrained-model with your specific changes denoted in the config;
                enc_model = AutoModel.from_pretrained(arch, config=enc_config)
            else:
                enc_model = AutoModel.from_pretrained(arch)
            # for records;
        
        # MLM
        elif task=='mlm' or task=='masked_language_modelling':
            # This is a generic model class that will be instantiated as one of the model classes of the library (with a masked language modeling head)
            if lang_model_config!=None:
                enc_config = update_huggingface_config(lang_model_config, enc_config)
                enc_model = AutoModelForMaskedLM.from_pretrained(arch, config=enc_config)
            else:
                enc_model = AutoModelForMaskedLM.from_pretrained(arch)

        # OTHERS TO BE IMPLEMNTED
        else:
            raise Exception('''the following task type is not defined for loading AutoModel{task} type. \n \\
                             Please refer to: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel \n \\
                             for more details''')
        
    else:
        # get the config;
        og_config = AutoConfig.from_pretrained(arch)
        if lang_model_config!=None:
            enc_config = update_huggingface_config(lang_model_config, og_config)
        else:
            print('using default config to initiate model (with random weights)')
            enc_config = og_config
        enc_model = AutoModel.from_config(enc_config)

    return enc_model, enc_config


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



def update_huggingface_config(new_config, og_config):
    if type(new_config)!=dict:
        new_config = OmegaConf.to_container(new_config)
        
    for k, _ in new_config.items():
        og_config.__dict__[k] = new_config[k]

    return og_config

def get_attn_config(config):
    language_config = config.model_config.language_encoder
    fusion_config = config.model_config.fusion
    model_config = config.model_config
    lang_encoder_name = language_config.name

    og_config = AutoConfig.from_pretrained(lang_encoder_name)
    og_config.__dict__['vocab_size'] =language_config["vocab_size"]
    og_config.__dict__['hidden_size']=model_config["hidden_size"]
    og_config.__dict__['num_hidden_layers']=fusion_config["num_layers"]
    og_config.__dict__['num_attention_heads']=fusion_config["num_heads"]
    og_config.__dict__['intermediate_size']=model_config["hidden_size"] * fusion_config["mlp_ratio"]
    og_config.__dict__['max_position_embeddings']=fusion_config["max_text_len"]
    og_config.__dict__['hidden_dropout_prob']=fusion_config["drop_rate"]
    og_config.__dict__['attention_probs_dropout_prob']=fusion_config["drop_rate"]
    
    return og_config

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def get_named_dict(model):
    named_layers = dict(model.named_modules())
    print(named_layers)
    return named_layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

if __name__=='__main__':
    import timm 

    model= build_image_encoder('resnet50', pretrained=True)