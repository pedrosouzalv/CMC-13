import pandas as pd
import numpy as np 

class Node():
    def __init__(self):
        self.value  = None
        self.next   = None
        self.childs = None
        
class ID3Classifier:
    def __init__(self):
        self.X          = None
        self.y          = None
        self.tree       = None
        self.idxs       = []
        self.features   = []
        self.categories = []   

    def _get_entropy(self, idxs):
        """
        Função para o cálculo da entropia.

        -> idxs: indices nos quais se deseja calcular a entropia (list of ints).
        
        -> retorno: entropia (float).
        """
        n = len(idxs)
        y = list(self.y['target'].loc[idxs])
        y_count = [y.count(x) for x in self.categories]
        entropy = sum([-count/n * np.log2(count/n) if count else 0 for count in y_count])
        return entropy

    def _get_info_gain(self, idxs, feature):
        """
        Função para o cálculo do ganho de informação.
        -> idxs: indices nos quais se deseja calcular o ganho de informação (list of ints).
        -> feature: atributo  para o qual se deseja calcular o ganho de informação (string).

        -> retorno: ganho de informação (float).
        """
        n = len(idxs)
        X_mask            = self.X[[feature]].loc[idxs]

        total_entropy     = self._get_entropy(idxs)        
        x_features        = list(X_mask[feature].values)     
        feature_vals      = list(set(x_features))       
        feature_v_count   = [x_features.count(x) for x in feature_vals]
        info_gain_feature = 0
        for i, x in  enumerate(feature_vals):
            idxs_mask = X_mask[X_mask[feature] == x].index.to_list()
            info_gain_feature +=  self._get_entropy(idxs_mask)*feature_v_count[i]/n
        return total_entropy - info_gain_feature

    def _get_feature_max_info_gain(self, idxs, features):
        """
        Função para obtenção do atributo com maior ganho de informação.

        -> idxs: indices nos quais se deseja obter o atributo com maior ganho de informação (list of ints).

        -> retorno: feature com maior ganho de informação (string).
        """
        features_info_gain = [self._get_info_gain(idxs, feature) for feature in features]
        return features[features_info_gain.index(max(features_info_gain))]

    def _make_tree(self, idxs, features, node):
        """
        Função para criação da árvore.

        -> idxs: indices nos quais se deseja ajustar a árvore (list of ints).
        -> features: atributos  nos quais se deseja ajustar a árvore (list of strings).
        -> node: nó da árvore (Node).

        -> retorno: nó raiz (Node).
        """
        if not node:
            node = Node() 
        y = self.y['target'].loc[idxs]
        
        #Apenas uma categoria alvo
        if len(set(y)) == 1:
            node.value = y.iloc[0]
            return node
        
        #Sem mais features para analisar
        if len(features) == 0:
            mode = y.mode()[0]
            node.value = mode 
            return node

        max_info_feature = self._get_feature_max_info_gain(idxs, features)
        node.value  = max_info_feature
        node.childs = []

        X_mask = self.X[max_info_feature].loc[idxs]
        feature_values = list(set(X_mask))

        for value in feature_values:
            child = Node()
            child.value = value  
            node.childs.append(child)  
            child_idx = X_mask[X_mask == value].index.to_list()
            if features and max_info_feature in features:
                features.remove(max_info_feature)
            child.next = self._make_tree(child_idx, features, child.next)
        return node

    def fit(self, X, y):
        """
        Função para ajustar a arvore aos dados de treino.

        -> X: DataFrame com features para treino (DataFrame).
        -> y: DataFrame com respostas (DataFrame).
        """
        self.X = X.reset_index(drop=True)
        self.y = pd.DataFrame(np.array(y), columns = ['target'])
        self.features = list(X.columns)
        self.categories = list(set(self.y['target']))
        self.idxs = list(range(self.X.shape[0]))
        self.tree = self._make_tree(self.idxs, self.features, self.tree)

    def _predict_instance(self, x):
        """
        Função para predição de classe de uma instância única.

        -> x: instância única (pd.Series).

        -> retorno: classe predita (class type).
        """
        eval_node = self.tree
        while eval_node.childs != None:
            feature = eval_node.value
            classe = x[feature]
            for child in eval_node.childs:
                if child.value == classe:
                    eval_node = child.next
        return eval_node.value
    
    def predict(self, X):
        """
        Função para predição.

        -> X: DataFrame com fetures de teste (DataFrame).

        -> retorno: classes preditas (np.array).
        """
        return X.apply(self._predict_instance,axis=1).values