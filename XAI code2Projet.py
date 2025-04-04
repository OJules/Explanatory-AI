# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AQ7Ve1XyxwaaQk1hT3M8JEvHOx5PiN_0

## Grad Cam pour analyse d'images
"""

# Imports pour Deep Learning
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Imports pour visualisation
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("TensorFlow version:", tf.__version__)

# Configuration du modèle (nous utiliserons ResNet50 au lieu de Xception)
model_builder = keras.applications.resnet50.ResNet50
img_size = (224, 224)  # Taille standard pour ResNet50
preprocess_input = keras.applications.resnet50.preprocess_input
decode_predictions = keras.applications.resnet50.decode_predictions

# Création du modèle avec poids ImageNet
model = model_builder(weights="imagenet")
print(model.summary())

# Définition de la dernière couche convolutive
last_conv_layer_name = "conv5_block3_out"
classifier_layer_names = [
    "avg_pool",
    "predictions",
]

def get_img_array(img_path, size):
    """Charge et prétraite l'image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, prediction_rank=1):
    """Génère la carte de chaleur Grad-CAM"""
    # Modèle pour la dernière couche convolutive
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Modèle pour le classificateur
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Calcul des gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        pred_index = tf.argsort(preds)[0][-prediction_rank]
        top_class_channel = preds[:, pred_index]

    # Gradients de la classe prédite par rapport à la dernière couche conv
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des canaux
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Génération de la heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Vous pouvez remplacer cette URL par celle de votre choix
img_path = keras.utils.get_file(
    "cat.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
)

# Affichage de l'image originale
display(Image(img_path))

# Préparation de l'image
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Prédictions
preds = model.predict(img_array)
print("Top 3 Prédictions:")
print(decode_predictions(preds, top=3)[0])

# Génération de la heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)

# Affichage de la heatmap seule
plt.figure(figsize=(8, 6))
plt.matshow(heatmap)
plt.title("Heatmap Grad-CAM")
plt.show()

# Chargement de l'image originale
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)

# Redimensionnement de la heatmap
heatmap = np.uint8(255 * heatmap)
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# Création de l'image avec heatmap colorée
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superposition
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Sauvegarde et affichage
save_path = "cat_gradcam.jpg"
superimposed_img.save(save_path)
display(Image(save_path))

"""## Interpretation de BERT"""

# Installation des packages nécessaires
!pip install transformers
!pip install captum
!pip install torch
!pip install seaborn
!pip install pandas
!pip install matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import json

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients

# Vérification du device disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Chargement du modèle BERT et du tokenizer
model_path = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

tokenizer = BertTokenizer.from_pretrained(model_path)

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask)
    return output.start_logits, output.end_logits

def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

def construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def get_topk_attributed_tokens(attrs, tokens, k=5):
    values, indices = torch.topk(attrs, k)
    top_tokens = [tokens[idx] for idx in indices]
    return top_tokens, values, indices

# Nouvel exemple : article médical sur le COVID-19
text = """
COVID-19 is caused by the SARS-CoV-2 virus. Common symptoms include fever, cough,
and loss of smell and taste. The virus primarily spreads through respiratory droplets
during close contact. Prevention measures include wearing masks, maintaining physical
distance, and frequent hand washing. Vaccines have been developed to protect against
severe illness.
"""

# Questions d'exemple
questions = [
    "What causes COVID-19?",
    "What are the main symptoms?",
    "How can we prevent COVID-19?"
]

# Tokens de référence
ref_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id
cls_token_id = tokenizer.cls_token_id

# Traitement de la première question
question = questions[0]
input_ids, ref_input_ids, sep_id = construct_input_ref_pair(question, text, ref_token_id, sep_token_id, cls_token_id)
attention_mask = construct_attention_mask(input_ids)

# Prédiction
start_scores, end_scores = predict(input_ids, attention_mask=attention_mask)

# Configuration de Layer Integrated Gradients
lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

# Calcul des attributions
attributions_start, delta_start = lig.attribute(
    inputs=input_ids,
    baselines=ref_input_ids,
    additional_forward_args=(None, None, attention_mask, 0),
    return_convergence_delta=True
)

attributions_end, delta_end = lig.attribute(
    inputs=input_ids,
    baselines=ref_input_ids,
    additional_forward_args=(None, None, attention_mask, 1),
    return_convergence_delta=True
)

# Création des visualisations
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().tolist())

start_position_vis = viz.VisualizationDataRecord(
    summarize_attributions(attributions_start),
    torch.max(torch.softmax(start_scores[0], dim=0)),
    torch.argmax(start_scores),
    torch.argmax(start_scores),
    "0",
    attributions_start.sum(),
    tokens,
    delta_start)

# Affichage des résultats
print('\nQuestion:', question)
print('Prédiction:', ' '.join(tokens[torch.argmax(start_scores):torch.argmax(end_scores)+1]))
print('\nVisualisation des attributions:')
viz.visualize_text([start_position_vis])

# Analyse par couche
def squad_pos_forward_func2(input_emb, attention_mask=None, position=0):
    pred = model(inputs_embeds=input_emb, attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

# Création des embeddings
input_embeddings = model.bert.embeddings(input_ids.long())
ref_input_embeddings = model.bert.embeddings(ref_input_ids.long())

layer_attrs_start = []
num_layers = model.config.num_hidden_layers

for i in range(num_layers):
    lc = LayerConductance(squad_pos_forward_func2, model.bert.encoder.layer[i])
    layer_attributions = lc.attribute(
        inputs=input_embeddings,
        baselines=ref_input_embeddings,
        additional_forward_args=(attention_mask, 0)
    )
    layer_attrs_start.append(summarize_attributions(layer_attributions).cpu().detach().numpy())

# Visualisation de la heatmap des attributions par couche
plt.figure(figsize=(15,5))
sns.heatmap(np.array(layer_attrs_start),
            xticklabels=tokens,
            yticklabels=range(1, num_layers+1),
            cmap='coolwarm')
plt.xlabel('Tokens')
plt.ylabel('Couches')
plt.title('Attribution par couche et token')
plt.show()

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    """
    Construit les token_type_ids pour l'input et la référence
    """
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    """
    Construit les position_ids pour l'input et la référence
    """
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids

def construct_whole_bert_embeddings(input_ids, ref_input_ids, token_type_ids=None,
                                  ref_token_type_ids=None, position_ids=None, ref_position_ids=None):
    """
    Construit les embeddings complets pour l'input et la référence
    """
    input_embeddings = model.bert.embeddings(input_ids.long(),
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids)
    ref_input_embeddings = model.bert.embeddings(ref_input_ids.long(),
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids)
    return input_embeddings, ref_input_embeddings

def pdf_attr(attrs, bins=100):
    """
    Calcule la densité de probabilité des attributions
    """
    return np.histogram(attrs, bins=bins, density=True)[0]

# Analyse détaillée d'un token spécifique
# Choisissons le token "SARS" comme exemple pour l'analyse
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
token_to_explain = tokens.index('sar') # Token pour SARS-CoV-2

layer_attrs_start_dist = []
layer_attrs_end_dist = []

# Collecte des attributions par couche
for i in range(model.config.num_hidden_layers):
    lc = LayerConductance(squad_pos_forward_func2, model.bert.encoder.layer[i])

    layer_attributions_start = lc.attribute(
        inputs=input_embeddings,
        baselines=ref_input_embeddings,
        additional_forward_args=(attention_mask, 0)
    )

    layer_attributions_end = lc.attribute(
        inputs=input_embeddings,
        baselines=ref_input_embeddings,
        additional_forward_args=(attention_mask, 1)
    )

    # Stockage des attributions pour le token spécifique
    layer_attrs_start_dist.append(layer_attributions_start[0, token_to_explain, :].cpu().detach().tolist())
    layer_attrs_end_dist.append(layer_attributions_end[0, token_to_explain, :].cpu().detach().tolist())

# Création des box plots pour visualiser la distribution des attributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Distribution des attributions de début
sns.boxplot(data=layer_attrs_start_dist, ax=ax1)
ax1.set_title(f"Distribution des attributions de début pour le token '{tokens[token_to_explain]}'")
ax1.set_xlabel('Couches')
ax1.set_ylabel('Attribution')

# Distribution des attributions de fin
sns.boxplot(data=layer_attrs_end_dist, ax=ax2)
ax2.set_title(f"Distribution des attributions de fin pour le token '{tokens[token_to_explain]}'")
ax2.set_xlabel('Couches')
ax2.set_ylabel('Attribution')

plt.tight_layout()
plt.show()

def pdf_attr(attrs, bins=100):
    return np.histogram(attrs, bins=bins, density=True)[0]

# Calcul des PDFs pour les attributions de fin
layer_attrs_end_pdf = map(lambda layer_attrs: pdf_attr(layer_attrs), layer_attrs_end_dist)
layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))

# Normalisation
attr_sum = np.array(layer_attrs_end_dist).sum(-1)
layer_attrs_end_pdf_norm = np.linalg.norm(layer_attrs_end_pdf, axis=-1, ord=1)
layer_attrs_end_pdf = np.transpose(layer_attrs_end_pdf)
layer_attrs_end_pdf = np.divide(layer_attrs_end_pdf, layer_attrs_end_pdf_norm,
                               where=layer_attrs_end_pdf_norm!=0)

# Visualisation des densités
plt.figure(figsize=(15, 8))
plt.plot(layer_attrs_end_pdf)
plt.title(f"Densité des attributions par couche pour le token '{tokens[token_to_explain]}'")
plt.xlabel('Bins')
plt.ylabel('Densité')
plt.legend(['Couche '+ str(i) for i in range(1, 13)])
plt.show()

# Calcul et visualisation de l'entropie
plt.figure(figsize=(15, 8))

# Éviter les log(0) en remplaçant les 0 par 1
layer_attrs_end_pdf[layer_attrs_end_pdf == 0] = 1
layer_attrs_end_pdf_log = np.log2(layer_attrs_end_pdf)

# Calcul de l'entropie
entropies = -(layer_attrs_end_pdf * layer_attrs_end_pdf_log).sum(0)

# Visualisation
plt.scatter(np.arange(12), attr_sum, s=entropies * 100)
plt.title(f"Attribution totale vs Entropie pour le token '{tokens[token_to_explain]}'")
plt.xlabel('Couches')
plt.ylabel('Attribution Totale')
plt.show()

# Affichage des statistiques
print(f"\nStatistiques pour le token '{tokens[token_to_explain]}':")
print(f"Entropie moyenne: {entropies.mean():.2f}")
print(f"Attribution totale moyenne: {attr_sum.mean():.2f}")

"""## réseaux de neurones graphiques"""

!pip install python-igraph

!pip install rdkit

import numpy as np
import torch
import igraph
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# 1. Fonctions utilitaires
def relevance_curves_xy_computation(nodes_x_coors_list, nodes_y_coors_list):
    relevance_line_curve_x_coors_list = np.array(nodes_x_coors_list)
    relevance_line_curve_y_coors_list = np.array(nodes_y_coors_list)

    if relevance_line_curve_x_coors_list[0] == relevance_line_curve_x_coors_list[1] == relevance_line_curve_x_coors_list[2] and \
            relevance_line_curve_y_coors_list[0] == relevance_line_curve_y_coors_list[1] == relevance_line_curve_y_coors_list[2]:
        relevance_line_curve_x_coors_list = relevance_line_curve_x_coors_list[0] + 0.1 * np.cos(np.linspace(0, 2 * np.pi, 50))
        relevance_line_curve_y_coors_list = relevance_line_curve_y_coors_list[0] + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 50))
    else:
        relevance_line_curve_x_coors_list = 0.75 * relevance_line_curve_x_coors_list + 0.25 * relevance_line_curve_x_coors_list.mean()
        relevance_line_curve_y_coors_list = 0.75 * relevance_line_curve_y_coors_list + 0.25 * relevance_line_curve_y_coors_list.mean()

        relevance_line_curve_x_coors_list = np.concatenate([
            np.linspace(relevance_line_curve_x_coors_list[0], relevance_line_curve_x_coors_list[0], 41),
            np.linspace(relevance_line_curve_x_coors_list[0], relevance_line_curve_x_coors_list[1], 20),
            np.linspace(relevance_line_curve_x_coors_list[1], relevance_line_curve_x_coors_list[2], 20),
            np.linspace(relevance_line_curve_x_coors_list[2], relevance_line_curve_x_coors_list[2], 41)])

        relevance_line_curve_y_coors_list = np.concatenate([
            np.linspace(relevance_line_curve_y_coors_list[0], relevance_line_curve_y_coors_list[0], 41),
            np.linspace(relevance_line_curve_y_coors_list[0], relevance_line_curve_y_coors_list[1], 20),
            np.linspace(relevance_line_curve_y_coors_list[1], relevance_line_curve_y_coors_list[2], 20),
            np.linspace(relevance_line_curve_y_coors_list[2], relevance_line_curve_y_coors_list[2], 41)])

        filt = np.exp(-np.linspace(-2, 2, 41) ** 2)
        filt = filt / filt.sum()

        relevance_line_curve_x_coors_list = np.convolve(relevance_line_curve_x_coors_list, filt, mode='valid')
        relevance_line_curve_y_coors_list = np.convolve(relevance_line_curve_y_coors_list, filt, mode='valid')

    return relevance_line_curve_x_coors_list, relevance_line_curve_y_coors_list

def set_graph_layout(adj_matrix: np.ndarray, seed):
    graph = igraph.Graph()
    graph.add_vertices(len(adj_matrix))
    graph.add_edges(zip(*np.where(adj_matrix == 1)))
    return np.array(list(graph.layout_kamada_kawai()))

def compute_walks(adj_matrix: np.ndarray):
    w = []
    for v1 in np.arange(len(adj_matrix)):
        for v2 in np.where(adj_matrix[v1])[0]:
            for v3 in np.where(adj_matrix[v2])[0]:
                w += [(v1, v2, v3)]
    return w

# 2. Classe GraphNet
class GraphNet:
    def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size: int):
        # Stockage des dimensions
        self.input_size = input_layer_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_layer_size

        # Initialisation des poids avec les dimensions correctes
        self.U = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, input_layer_size ** -.5,
                                                    [input_layer_size, hidden_layer_size])))
        self.W1 = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, hidden_layer_size ** -.5,
                                                     [hidden_layer_size, hidden_layer_size])))
        self.W2 = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, hidden_layer_size ** -.5,
                                                     [hidden_layer_size, hidden_layer_size])))
        self.V = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, hidden_layer_size ** -.5,
                                                    [hidden_layer_size, output_layer_size])))

        self.params = [self.U, self.W1, self.W2, self.V]

    def _pad_input(self, X):
        """Ajuste les dimensions de l'entrée si nécessaire"""
        batch_size = X.shape[0]
        if batch_size < self.input_size:
            padding = torch.zeros((batch_size, self.input_size - batch_size))
            return torch.cat([X, padding], dim=1)
        return X

    def forward_pass(self, adj_matrix: torch.Tensor):
        # Créer la matrice identité et ajuster les dimensions
        batch_size = len(adj_matrix)
        H = self._pad_input(torch.eye(batch_size))

        # Forward pass
        H = H.matmul(self.U).clamp(min=0)
        H = (adj_matrix.transpose(1, 0).matmul(H.matmul(self.W1))).clamp(min=0)
        H = (adj_matrix.transpose(1, 0).matmul(H.matmul(self.W2))).clamp(min=0)
        H = H.matmul(self.V).clamp(min=0)

        return H.mean(dim=0)

    def lrp_computation(self, adj_matrix: torch.Tensor, gamma: float, target: int, indexes: tuple):
        if indexes is not None:
            j, k = indexes
            M_j = torch.FloatTensor(np.eye(len(adj_matrix))[j][:, np.newaxis])
            M_k = torch.FloatTensor(np.eye(len(adj_matrix))[k][:, np.newaxis])

        # Poids modifiés pour LRP
        W1p = self.W1 + gamma * self.W1.clamp(min=0)
        W2p = self.W2 + gamma * self.W2.clamp(min=0)
        Vp = self.V + gamma * self.V.clamp(min=0)

        # Initialisation avec les bonnes dimensions
        X = torch.eye(len(adj_matrix))
        X = self._pad_input(X)
        X.requires_grad_(True)

        # Calcul LRP
        H = X.matmul(self.U).clamp(min=0)

        P = adj_matrix.transpose(1, 0).matmul(H.matmul(self.W1))
        Pt = adj_matrix.transpose(1, 0).matmul(H.matmul(W1p))
        Qt = (Pt * (P / (Pt + 1e-6)).data).clamp(min=0)

        if indexes is not None:
            H = Qt * M_j + (1 - M_j) * (Qt.data)

        P = adj_matrix.transpose(1, 0).matmul(H.matmul(self.W2))
        Pt = adj_matrix.transpose(1, 0).matmul(H.matmul(W2p))
        Qt = (Pt * (P / (Pt + 1e-6)).data).clamp(min=0)

        if indexes is not None:
            H = Qt * M_k + (1 - M_k) * (Qt.data)

        P = H.matmul(self.V)
        Pt = H.matmul(Vp)
        Qt = (Pt * (P / (Pt + 1e-6)).data).clamp(min=0)

        Y = Qt.mean(dim=0)[target]
        Y.backward()

        return X.data * X.grad

# 3. Fonctions de visualisation et analyse
def explain_graph_LRP(input_graph, nn, target, gamma=None, ax=None):
    r = input_graph['layout']
    r = r - r.min(axis=0)
    r = r / r.max(axis=0) * 2 - 1

    N = len(input_graph['adjacency'])
    for i in np.arange(N):
        for j in np.arange(N):
            if input_graph['adjacency'][i, j] > 0 and i != j:
                plt.plot([r[i, 0], r[j, 0]], [r[i, 1], r[j, 1]],
                        color='gray', lw=0.5, ls='dotted')
    ax.plot(r[:, 0], r[:, 1], 'o', color='black', ms=3)

    for (i, j, k) in input_graph['walks']:
        R = nn.lrp_computation(input_graph['laplacian'], gamma, target, (j, k))[i].sum()
        tx, ty = relevance_curves_xy_computation([r[i, 0], r[j, 0], r[k, 0]],
                                               [r[i, 1], r[j, 1], r[k, 1]])

        if R > 0.0:
            alpha = np.clip(20 * R.data.numpy(), 0, 1)
            ax.plot(tx, ty, alpha=alpha, color='red', lw=1.2)
        if R < -0.0:
            alpha = np.clip(-20 * R.data.numpy(), 0, 1)
            ax.plot(tx, ty, alpha=alpha, color='blue', lw=1.2)

def analyze_molecular_relevances(mol_graphs, model, gamma=0.1):
    plt.figure(figsize=(15, 4))

    for idx, graph in enumerate(mol_graphs):
        ax = plt.subplot(1, len(mol_graphs), idx + 1)
        explain_graph_LRP(graph, model, graph['target'], gamma=gamma, ax=ax)
        plt.title(f"Molécule {idx+1}\nRelevances pour target={graph['target']}")

    plt.tight_layout()
    plt.show()

# 4. Préparation des données et entraînement
def molecule_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    adj_matrix = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i,j] = 1
        adj_matrix[j,i] = 1

    return {
        'adjacency': torch.FloatTensor(adj_matrix),
        'laplacian': torch.FloatTensor(adj_matrix/(np.outer(adj_matrix.sum(axis=1),
                                                          adj_matrix.sum(axis=1))**.5+1e-9)),
        'target': 1 if mol.GetNumAtoms() > 10 else 0,
        'layout': set_graph_layout(adj_matrix, 42),
        'walks': compute_walks(adj_matrix)
    }

molecules = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirine
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caféine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testostérone
    'CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1'  # Salbutamol
]

mol_graphs = [molecule_to_graph(smiles) for smiles in molecules]

def train_molecular_gnn(mol_graphs, hidden_size=64, epochs=1000):
    max_nodes = max(g['adjacency'].shape[0] for g in mol_graphs)
    model = GraphNet(max_nodes, hidden_size, 2)

    # Changement de l'optimiseur et ajout d'un learning rate plus agressif
    optimizer = torch.optim.Adam(model.params, lr=0.01)

    # Ajout d'un scheduler pour ajuster le learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    print("Entraînement du modèle:")
    print("   Epoch | Loss")
    print("   ------------")

    for epoch in range(epochs):
        total_loss = 0
        for graph in mol_graphs:
            optimizer.zero_grad()

            output = model.forward_pass(graph['laplacian'])
            # Modification de la cible pour éviter la saturation
            target = torch.tensor([0.9 if graph['target'] == 1 else 0.1,
                                 0.9 if graph['target'] == 0 else 0.1])

            # Utilisation de la BCE Loss au lieu de MSE
            loss = torch.nn.functional.binary_cross_entropy(
                torch.sigmoid(output),
                target
            )

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Mise à jour du scheduler
        scheduler.step(total_loss)

        if epoch % 100 == 0:
            print(f"   {epoch:5d} | {total_loss/len(mol_graphs):.4f}")

    return model

model = train_molecular_gnn(mol_graphs)

analyze_molecular_relevances(mol_graphs, model)