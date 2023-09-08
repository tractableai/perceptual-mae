import torch
import torch.nn as nn
import torch.nn.functional as F

class AggregatedAttentionHead(nn.Module):
    def __init__(self, config):
        super(AggregatedAttentionHead, self).__init__()
        self.config = config
        self.dataset_config = config.dataset_config
        self.model_config= config.model_config

        # pre-image classifier
        classifier_layers=[]
        for i in range(self.dataset_config.max_images):
            classifier_layers.append(nn.Linear(self.model_config.classifier.in_dim, 
                                                self.model_config.classifier.num_classes))
    
        self.per_image_classifier= nn.ModuleList(classifier_layers)
        # attention aggregator;
        self.query_layer= nn.Linear(self.model_config.classifier.num_classes, self.model_config.attention.in_dim)
        self.key_layer= nn.Linear(self.model_config.classifier.num_classes, self.model_config.attention.in_dim)
        self.value_layer= nn.Linear(self.model_config.classifier.num_classes, self.model_config.attention.in_dim)
        self.attention_layer = nn.MultiheadAttention(self.model_config.attention.in_dim, self.model_config.attention.num_heads, batch_first=True)
        # final classifier;
        self.final_classifier= nn.Linear(self.model_config.classifier.in_dim * self.dataset_config.max_images, 
                                         self.model_config.classifier.num_classes)

    def forward(self, x):
         # x: B x S x dim
        outputs=[]
        for i in range(x.size(1)):
            pre_classfication_logits = self.per_image_classifier[i](x[:,i,:])
            outputs.append(pre_classfication_logits)
        
        outputs = torch.stack(outputs).squeeze().permute(1,0,2)
        # calculate attention on each classified output;
        query= self.query_layer(outputs)
        key= self.key_layer(outputs)
        value= self.value_layer(outputs)
        # attention;
        attn_out, attn_output_weights = self.attention_layer(query, key, value)
        # final classifier;
        flattened = attn_out.contiguous().view(x.size(0), -1)

        logits = self.final_classifier(flattened)
        return logits



"""
code taken from: https://github.com/AMLab-Amsterdam/AttentionDeepMIL

ref: Attention-based Deep Multiple Instance Learning
     https://arxiv.org/pdf/1802.04712.pdf
"""
class Attention_OutputHead(nn.Module):
    def __init__(self, model_config):
        super(Attention_OutputHead, self).__init__()
        self.model_config= model_config.classifier
        self.L = self.model_config.L #500
        self.D = self.model_config.D #128
        self.K = self.model_config.K #1
        # feature extractor removed from original. Will be using our own;

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention_OutputHead(nn.Module):
    def __init__(self, model_config):
        super(GatedAttention_OutputHead, self).__init__()
        self.model_config= model_config.classifier
        self.L = self.model_config.L #500
        self.D = self.model_config.D #128
        self.K = self.model_config.K #1

        # feature extractor removed from original. Will be using our own;

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A