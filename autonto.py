#General Imports
import os
import math
import operator
import argparse
from collections import defaultdict
import argparse
import subprocess
from math import log


#Imports for clustering phase
from scipy.spatial.distance import cosine
import warnings
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import _k_means_fast as _k_means
from sklearn.cluster.k_means_ import (
    _check_sample_weight,
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _validate_center_shape,
)
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import _num_samples

# Imports for local embedding phase
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Helper Functions
def load_embeddings(embedding_file):
	if embedding_file is None:
		return {}
	word_to_vec = {}
	with open(embedding_file, 'r') as fin:
		header = fin.readline()
		for line in fin:
			items = line.strip().split()
			word = items[0]
			vec = [float(v) for v in items[1:]]
			word_to_vec[word] = vec
	return word_to_vec

def kl_divergence(p, q):
	if len(p) != len(q):
		print('KL divergence error: p, q have different length')
	c_entropy = 0
	for i in range(len(p)):
		if p[i] > 0:
			c_entropy += p[i] * math.log(float(p[i]) / q[i])
	return c_entropy

def avg_weighted_colors(color_list, c_size):
	# print color_list
	# given a weighted color list, return the result
	result_color = [0] * c_size
	for (color, weight) in color_list:
		w_color = [x * weight for x in color]
		# print w_color
		result_color = map(operator.add, result_color, w_color)
		# print result_color
	return l1_normalize(result_color)


def l1_normalize(p):
	sum_p = sum(p)
	if sum_p <= 0:
		print('Normalizing invalid distribution')
	return [x/sum_p for x in p]

def cossim(p, q):
	if len(p) != len(q):
		print('KL divergence error: p, q have different length')
	
	p_len = q_len = mix_len = 0

	for i in range(len(p)):
		mix_len += p[i] * q[i]
		p_len += p[i] * p[i]
		q_len += q[i] * q[i]

	return mix_len / (math.sqrt(p_len) * math.sqrt(q_len))

def euclidean_distance(p, q):
	if len(p) != len(q):
		print ('Euclidean distance error: p, q have different length')
	
	distance = 0

	for i in range(len(p)):
		distance += math.pow(p[i] - q[i], 2)

	return math.sqrt(distance)


def euclidean_cluster(ps, c):
	if len(ps) == 0 or c == None:
		print ('Cluster is empty')

	distance = 0

	for p in ps:
		for i in range(len(p)):
			distance += math.pow(p[i] - c[i], 2)
	distance /= len(ps)

	return math.sqrt(distance)


def dot_product(p, q):
	if len(p) != len(q):
		print ('KL divergence error: p, q have different length')
	
	p_len = q_len = mix_len = 0

	for i in range(len(p)):
		mix_len += p[i] * q[i]

	return mix_len

def softmax(score_list):
	# normalization of exp
	exp_sum = 0
	for score in score_list:
		exp_sum += math.exp(score)

	exp_list = []
	for score in score_list:
		normal_value = math.exp(score) / exp_sum
		exp_list.append(normal_value)
	return exp_list


def softmax_for_map(t_map):
	exp_sum = 0
	for key in t_map:
		score = t_map[key]
		exp_sum += math.exp(score)

	for key in t_map:
		score = t_map[key]
		normal_value = math.exp(score) / exp_sum
		t_map[key] = normal_value


def avg_emb_with_distinct(ele_map, embs_from, dist_map, vec_size):

	avg_emb = [0] * vec_size
	t_weight = 0

	for key, value in ele_map.iteritems():
		t_emb = embs_from[key]
		w = value * dist_map[key]
		for i in range(vec_size):
			avg_emb[i] += w * t_emb[i]
		t_weight += w

	for i in range(vec_size):
		avg_emb[i] /= t_weight

	return avg_emb


def avg_emb(ele_map, embs_from, vec_size):
	
	avg_emb = [0] * vec_size
	t_weight = 0

	for key, value in ele_map.iteritems():
		t_emb = embs_from[key]
		w = value
		for i in range(vec_size):
			avg_emb[i] += w * t_emb[i]
		t_weight += w

	for i in range(vec_size):
		avg_emb[i] /= t_weight

	return avg_emb

def load_hier_f(hier_f):
	hier_map = {}

	with open(hier_f) as f:
		idx = 0
		for line in f:
			topic = line.split()[0]
			hier_map[topic] = idx
			idx += 1

	return hier_map

# ensure the path for the output file exist
def ensure_directory_exist(file_name):
	directory = os.path.dirname(file_name)
	if not os.path.exists(directory):
		os.makedirs(directory)

# IO methods
# the complete data set
class DataSet:

    def __init__(self, embedding_file, document_file):
        self.documents = self.load_documents(document_file)
        self.embeddings = self.load_embeddings(embedding_file)
        # the initial complete set of keywords
        # self.keywords = self.load_keywords(candidate_file)
        # self.keyword_set = set(self.keywords)
        # self.documents_trimmed = self.get_trimmed_documents(self.documents, self.keyword_set)
        # assert len(self.documents) == len(self.documents_trimmed)

    def load_embeddings(self, embedding_file):
        if embedding_file is None:
            return {}
        word_to_vec = {}
        with open(embedding_file, 'r') as fin:
            header = fin.readline()
            for line in fin:
                items = line.strip().split()
                word = items[0]
                vec = [float(v) for v in items[1:]]
                word_to_vec[word] = vec
        return word_to_vec

    def load_documents(self, document_file):
        documents = []
        with open(document_file, 'r') as fin:
            for line in fin:
                keywords = line.strip().split()
                documents.append(keywords)
        print('Length in Dataset.load_documents - ',len(documents))
        return documents


# sub data set for each cluster
class SubDataSet:

    def __init__(self, full_data, doc_id_file, keyword_file):
        self.keywords = self.load_keywords(keyword_file, full_data)
        self.keyword_to_id = self.gen_keyword_id()
        self.keyword_set = set(self.keywords)
        self.embeddings = self.load_embeddings(full_data)
        self.documents, self.original_doc_ids = self.load_documents(full_data, doc_id_file)
        self.keyword_idf = self.build_keyword_idf()

    def load_keywords(self, keyword_file, full_data):
        keywords = []
        with open(keyword_file, 'r') as fin:
            for line in fin:
                keyword = line.strip()
                if keyword in full_data.embeddings:
                    keywords.append(keyword)
                else:
                    print(keyword, ' not in the embedding file')
        return keywords

    def gen_keyword_id(self):
        keyword_to_id = {}
        for idx, keyword in enumerate(self.keywords):
            keyword_to_id[keyword] = idx
        return keyword_to_id

    def load_embeddings(self, full_data):
        embeddings = full_data.embeddings
        ret = []
        for word in self.keywords:
            vec = embeddings[word]
            ret.append(vec)
        return np.array(ret)

    def load_documents(self, full_data, doc_id_file):
        '''
        :param full_data:
        :param doc_id_file:
        :return: trimmed documents along with its corresponding ids
        '''
        doc_ids = self.load_doc_ids(doc_id_file)
        trimmed_doc_ids, trimmed_docs = [], []
        #print(len(full_data.documents))
       
        for doc_id in doc_ids:
            #print(doc_id)
            doc = full_data.documents[doc_id]
            trimmed_doc = [e for e in doc if e in self.keyword_set]
            if len(trimmed_doc) > 0:
                trimmed_doc_ids.append(doc_id)
                trimmed_docs.append(trimmed_doc)
        return trimmed_docs, trimmed_doc_ids

    def load_doc_ids(self, doc_id_file):
        doc_ids = []
        with open(doc_id_file, 'r') as fin:
            for line in fin:
                doc_id = int(line.strip())
                doc_ids.append(doc_id)
        return doc_ids

    def build_keyword_idf(self):
        keyword_idf = defaultdict(float)
        for doc in self.documents:
            word_set = set(doc)
            for word in word_set:
                if word in self.keyword_set:
                    keyword_idf[word] += 1.0
        N = len(self.documents)
        for w in keyword_idf:
            keyword_idf[w] = log(1.0 + N / keyword_idf[w])
        return keyword_idf

    # output_file: one integrated file;
    def write_cluster_members(self, clus, cluster_file, parent_dir):
        n_cluster = clus.n_cluster
        clusters = clus.clusters  # a dict: cluster id -> keywords
        with open(cluster_file, 'w') as fout:
            for clus_id in range(n_cluster):
                members = clusters[clus_id]
                for keyword_id in members:
                    keyword = self.keywords[keyword_id]
                    fout.write(str(clus_id) + '\t' + keyword + '\n')
        # write the cluster for each sub-folder
        clus_centers = clus.center_ids
        for clus_id, center_keyword_id in clus_centers:
            center_keyword = self.keywords[center_keyword_id]
            output_file = parent_dir + center_keyword + '/seed_keywords.txt'
            ensure_directory_exist(output_file)
            members = clusters[clus_id]
            with open(output_file, 'w') as fout:
                for keyword_id in members:
                    keyword = self.keywords[keyword_id]
                    fout.write(keyword + '\n')

    def write_cluster_centers(self, clus, parent_description, output_file):
        clus_centers = clus.center_ids
        center_names = []
        with open(output_file, 'w') as fout:
            for cluster_id, keyword_idx in clus_centers:
                keyword = self.keywords[keyword_idx]
                center_names.append(keyword)
                fout.write(keyword + ' ' + parent_description + '\n')
        return center_names


    def write_document_membership(self, clus, output_file, parent_dir):
        n_cluster = clus.n_cluster
        keyword_membership = clus.membership  # an array containing the membership of the keywords
        cluster_document_map = defaultdict(list)  # key: cluster id, value: document list
        with open(output_file, 'w') as fout:
            for idx, doc in zip(self.original_doc_ids, self.documents):
                doc_membership = self.get_doc_membership(n_cluster, doc, keyword_membership)
                cluster_id = self.assign_document(doc_membership)
                cluster_document_map[cluster_id].append(idx)
                fout.write(str(idx) + '\t' + str(cluster_id) + '\n')
        # write the document ids for each sub-folder
        clus_centers = clus.center_ids
        for clus_id, center_keyword_id in clus_centers:
            center_keyword = self.keywords[center_keyword_id]
            output_file = parent_dir + center_keyword + '/doc_ids.txt'
            ensure_directory_exist(output_file)
            doc_ids = cluster_document_map[clus_id]
            with open(output_file, 'w') as fout:
                for doc_id in doc_ids:
                    fout.write(str(doc_id) + '\n')


    def get_doc_membership(self, n_cluster, document, keyword_membership):
        ret = [0.0] * n_cluster
        ## Strength of each document on each cluster is the tf-idf score. The tf part is considered during the
        ## enumeration of document tokens.
        for keyword in document:
            keyword_id = self.keyword_to_id[keyword]
            cluster_id = keyword_membership[keyword_id]
            idf = self.keyword_idf[keyword]
            ret[cluster_id] += idf
        return ret

    def assign_document(self, doc_membership):
        ## Currently document cluster is a hard partition.
        best_idx, max_score = -1, 0
        for idx, score in enumerate(doc_membership):
            if score > max_score:
                best_idx, max_score = idx, score
        return best_idx

# Keyphrase ranking functions

def read_caseolap_result(case_file):
	phrase_map = {}
	cell_map = {}

	cell_cnt = 0
	with open(case_file) as f:
		for line in f:
			cell_cnt += 1
			segments = line.strip('\r\n ').split('\t')
			cell_id, phs_str = segments[0], segments[1][1:-1]
			cell_map[cell_id] = []
			segments = phs_str.split(', ')
			for ph_score in segments:
				parts = ph_score.split('|')
				ph, score = parts[0], float(parts[1])
				if ph not in phrase_map:
					phrase_map[ph] = {}
				phrase_map[ph][cell_id] = score
				cell_map[cell_id].append((ph, score))

	return phrase_map, cell_map, cell_cnt


def rank_phrase(case_file):

	ph_dist_map = {}
	smoothing_factor = 0.0

	phrase_map, cell_map, cell_cnt = read_caseolap_result(case_file)

	unif = [1.0 / cell_cnt] * cell_cnt

	for ph in phrase_map:
		ph_vec = [x[1] for x in phrase_map[ph].items()]
		if len(ph_vec) < cell_cnt:
			ph_vec += [0] * (cell_cnt - len(ph_vec))
		# smoothing
		ph_vec = [x + smoothing_factor for x in ph_vec]
		ph_vec = l1_normalize(ph_vec)
		ph_dist_map[ph] = kl_divergence(ph_vec, unif)

	ranked_list = sorted(ph_dist_map.items(), key=operator.itemgetter(1), reverse=True)
	
	return ranked_list


def write_keywords(o_file, ranked_list, thres):
	with open(o_file, 'w+') as g:
		for ph in ranked_list:
			if ph[1] > thres:
				g.write('%s\n' % (ph[0]))
	tmp_file = o_file + '-score.txt'
	with open(tmp_file, 'w+') as g:
		for ph in ranked_list:
			g.write('%s\t%f\n' % (ph[0], ph[1]))

def main_rank_phrase(input_f, output_f, thres):
  ranked_list = rank_phrase(input_f)
  write_keywords(output_f, ranked_list, thres)
  print("[CaseOLAP] Finish pushing general terms up")

  from heapq import heappush, heappop, heappushpop, nsmallest, nlargest
import codecs
import math
import ast
import argparse
import copy


class CaseSlim:

	def bm25_df_paper(self, df, max_df, tf, dl, avgdl, k=1.2, b=0.5, multiplier=3):
		score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
		df_factor = math.log(1 + df, 2) / math.log(1 + max_df, 2)
		score *= df_factor
		score *= multiplier
		return score


	def softmax_paper(self, score_list):
		# normalization of exp
		exp_sum = 1
		for score in score_list:
			exp_sum += math.exp(score)

		exp_list = []
		for score in score_list:
			normal_value = math.exp(score) / exp_sum
			exp_list.append(normal_value)
		return exp_list


	def compute(self, score_type='ALL'):
		'''
		-- score_type --
			ALL: all three factors
			POP: only popularity
			DIS: only distinctive
			INT: only integrity
			NOPOP: no populairty
			NODIS: no distinctive
			NOINT: no integrity
		'''
		scores = {}
		multiplier = 1

		
		sum_self = self.sum_cnt
		num_context_cells = len(self.sum_cnt_context) + 1
		total_sum = sum(self.sum_cnt_context.values()) + sum_self
		avgdl = total_sum / float(num_context_cells)

		# method 1
		for phrase in self.phrase_cnt:
			lower_phrase = phrase.lower()
			score = 1
			nor_phrase = self.normalize(lower_phrase)
			self_cnt = self.phrase_cnt[phrase]
			self_df = self.phrase_df[phrase]
			
			group = [(self_df, self.max_df, self_cnt, sum_self)]

			self.context_groups[phrase] = []
			for phrase_group, phrase_values in self.phrase_cnt_context.items():
				context_df = self.phrase_df_context[phrase_group].get(phrase, 0)
				sum_context = self.sum_cnt_context[phrase_group]
				context_cnt = phrase_values.get(phrase, 0)
				maxdf_context = self.max_df_context[phrase_group]

				if (context_cnt > 0):
					group.append((context_df, maxdf_context, context_cnt, sum_context))
					self.context_groups[phrase].append((context_df, maxdf_context, context_cnt, sum_context))
				
			score_list = []
			for record in group:
				score_list.append(self.bm25_df_paper(record[0], record[1], record[2], record[3], avgdl))
			distinct = self.softmax_paper(score_list)[0]
			
			popularity = math.log(1 + self_df, 2)
			try:
				integrity = float(self.global_scores[nor_phrase])
			except:
				integrity = 0.8

			if score_type == 'ALL':
				score = distinct * popularity * integrity
			elif score_type == 'POP':
				score = popularity
			elif score_type == 'DIS':
				score = distinct
			elif score_type == 'INT':
				score = integrity
			elif score_type == 'NOPOP':
				score = distinct * integrity
			elif score_type == 'NODIS':
				score = popularity * integrity
			elif score_type == 'NOINT':
				score = popularity * distinct
			else:
				score = 0

			scores[phrase] = score

		ranked_list = [(phrase, scores[phrase]) for phrase in sorted(scores, key=scores.get, reverse=True)]
		
		return ranked_list


	def agg_phrase_cnt_df(self, freq_data, selected_docs = None):
		phrase_cnt = {}
		phrase_df = {}

		if selected_docs == None:
			for doc_index in freq_data:
				for phrase in freq_data[doc_index]:
					if phrase not in phrase_cnt:
						phrase_cnt[phrase] = 0
					phrase_cnt[phrase] += freq_data[doc_index][phrase]
		else:
			for doc_index in selected_docs:
				for phrase in freq_data.get(doc_index, {}):
					if phrase not in phrase_cnt:
						phrase_cnt[phrase] = 0
					if phrase not in phrase_df:
						phrase_df[phrase] = 0
					phrase_cnt[phrase] += freq_data[doc_index][phrase]
					phrase_df[phrase] += 1

		return phrase_cnt, phrase_df


	def normalize(self, word):
		word = word.lower()
		result = []
		for i in range(len(word)):
			if word[i].isalpha() or word[i] == '\'':
				result.append(word[i])
			else:
				result.append(' ')
		word = ''.join(result)
		return ' '.join(word.split())


	def __init__(self, freq_data, selected_docs, context_doc_groups, global_scores=None):
		# print 'handle slim version'
		self.phrase_cnt, self.phrase_df = self.agg_phrase_cnt_df(freq_data, selected_docs)
		self.phrase_cnt_context = {}
		self.phrase_df_context = {}
		if len(self.phrase_df) > 0:
			self.max_df = max(self.phrase_df.values())
		else:
			self.max_df = 0
		self.max_df_context = {}
		self.dc_context = {}
		self.self_dc = len(selected_docs)
		self.sum_cnt = sum(self.phrase_cnt.values())
		self.sum_cnt_context = {}
		self.global_scores = global_scores
		for group, docs in context_doc_groups.items():
			self.phrase_cnt_context[group], self.phrase_df_context[group] = self.agg_phrase_cnt_df(freq_data, docs)
			if len(self.phrase_df_context[group]) > 0:
				self.max_df_context[group] = max(self.phrase_df_context[group].values())
			else:
				self.max_df_context[group] = 0
			self.dc_context[group] = len(docs)
			self.sum_cnt_context[group] = sum(self.phrase_cnt_context[group].values())

		# added for exploration
		self.context_groups = {}
		self.ranked_list = []


def read_data(label_f, link_f):
  '''

  :param label_f: doc_membership_file
  :param link_f: keyword_cnt, <doc_id>\t<word1>\t<count1>\t<word2>\t<count2>
  :return:
   cells: key: cell_id (int), value: doc_id_list
   freq_data: key: doc_id, value: a dict (key: phrase, value: phrase count)
   phrases: a set of phrases
  '''

  cells = {}
  freq_data = {}
  docs = set()
  phrases = set()

  with open(label_f, 'r+') as f:
    for line in f:
      segments = line.strip('\n\r').split('\t')
      cell = segments[1]
      doc_id = segments[0]
      if cell not in cells:
        cells[cell] = []
      cells[cell].append(doc_id)
      docs.add(doc_id)

  print('[CaseOLAP] Read document cluster membership file done.')

  with open(link_f, 'r+') as f:
    for line in f:
      segments = line.strip('\n\r ').split('\t')
      doc_id = segments[0]
      if doc_id not in docs:
        continue
      if doc_id not in freq_data:
        freq_data[doc_id] = {}

      for i in range(1, len(segments), 2):
        phrase, w = segments[i], int(segments[i+1])
        phrases.add(phrase)
        freq_data[doc_id][phrase] = w

  print('[CaseOLAP] Read keyword_cnt file done.')

  return cells, freq_data, phrases


def read_target_tokens(token_f):
  '''
  :param token_f: cluster_keyword_file
  :return:
  '''

  tokens = set()
  with open(token_f, 'r+') as f:
    for line in f:
      segments = line.strip('\r\n ').split('\t')
      tokens.add(segments[1])

  print('[CaseOLAP] Read keyword cluster membership file done.')
  return tokens


def run_caseolap(cells, freq_data, target_phs, o_file, verbose=3, top_k=200):
  of = open(o_file, 'w+')

  for cell in cells:
    print('[CaseOLAP] Running CaseOLAP for cell: %s' % cell)

    selected_docs = cells[cell]
    context_doc_groups = copy.copy(cells)
    context_doc_groups.pop(cell, None)
    caseslim = CaseSlim(freq_data, selected_docs, context_doc_groups)

    top_phrases = caseslim.compute(score_type="NOINT")
    of.write('%s\t' % cell)

    phr_str = ', '.join([ph[0] + '|' + str(ph[1]) for ph in top_phrases if ph[0] in target_phs])
    of.write('[%s]\n' % phr_str)
    print('[CaseOLAP] Finished CaseOLAP for cell: %s' % cell)


def main_caseolap(link_f, cell_f, token_f, output_f):
	cells, freq_data, phrases = read_data(cell_f, link_f)
	target_phs = read_target_tokens(token_f)
	run_caseolap(cells, freq_data, target_phs, output_f)


### Spherical Clustering

def _spherical_kmeans_single_lloyd(
    X,
    n_clusters,
    sample_weight=None,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    precompute_distances=True,
):
    """
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    """
    random_state = check_random_state(random_state)

    sample_weight = _check_sample_weight(sample_weight, X)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment
        # TODO: _labels_inertia should be done with cosine distance
        #       since ||a - b|| = 2(1 - cos(a,b)) when a,b are unit normalized
        #       this doesn't really matter.
        labels, inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

        # computation of the means
        if sp.issparse(X):
            centers = _k_means._centers_sparse(
                X, sample_weight, labels, n_clusters, distances
            )
        else:
            centers = _k_means._centers_dense(
                X.astype(float),
                sample_weight.astype(float),
                labels,
                n_clusters,
                distances.astype(float),
            )

        # l2-normalize centers (this is the main contibution here)
        centers = normalize(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            best_centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

    return best_labels, best_inertia, best_centers, i + 1


def spherical_k_means(
    X,
    n_clusters,
    sample_weight=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    algorithm="auto",
    return_n_iter=False,
):
    """Modified from sklearn.cluster.k_means_.k_means.
    """
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    best_inertia = np.infty
    # avoid forcing order when copy_x=False
    order = "C" if copy_x else None
    X = check_array(
        X, accept_sparse="csr", dtype=[np.float64, np.float32], order=order, copy=copy_x
    )
    # verify that the number of samples given is larger than k
    if _num_samples(X) < n_clusters:
        raise ValueError(
            "n_samples=%d should be >= n_clusters=%d" % (_num_samples(X), n_clusters)
        )
    tol = _tolerance(X, tol)

    if hasattr(init, "__array__"):
        init = check_array(init, dtype=X.dtype.type, order="C", copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d" % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = _spherical_kmeans_single_lloyd(
                X,
                n_clusters,
                sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
            )

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(
                X,
                n_clusters,
                sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed,
            )
            for seed in seeds
        )

        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class SphericalKMeans(KMeans):
    """Spherical K-Means clustering

    Modfication of sklearn.cluster.KMeans where cluster centers are normalized
    (projected onto the sphere) in each iteration.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        n_jobs=1,
        verbose=0,
        random_state=None,
        copy_x=True,
        normalize=True,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.normalize = normalize

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """
        if self.normalize:
            X = normalize(X)

        random_state = check_random_state(self.random_state)

        # TODO: add check that all data is unit-normalized

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = spherical_k_means(
            X,
            n_clusters=self.n_clusters,
            sample_weight=sample_weight,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            tol=self.tol,
            random_state=random_state,
            copy_x=self.copy_x,
            n_jobs=self.n_jobs,
            return_n_iter=True,
        )

        return self



class Clusterer:

    def __init__(self, data, n_cluster):
        self.data = data
        self.n_cluster = n_cluster
        self.clus = SphericalKMeans(n_cluster)
        self.clusters = defaultdict(list)  # cluster id -> members
        self.membership = None  # a list contain the membership of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers and keyword IDs
        self.inertia_scores = None

    def fit(self):
        self.clus.fit(self.data)
        labels = self.clus.labels_
        for idx, label in enumerate(labels):
            self.clusters[label].append(idx)
        self.membership = labels
        self.center_ids = self.gen_center_idx()
        self.inertia_scores = self.clus.inertia_
        print('Clustering concentration score:', self.inertia_scores)

    # find the idx of each cluster center
    def gen_center_idx(self):
        ret = []
        for cluster_id in range(self.n_cluster):
            center_idx = self.find_center_idx_for_one_cluster(cluster_id)
            ret.append((cluster_id, center_idx)) #(clusterID,KeywordID)
        return ret


    def find_center_idx_for_one_cluster(self, cluster_id):
        query_vec = self.clus.cluster_centers_[cluster_id]
        members = self.clusters[cluster_id]
        best_similarity, ret = -1, -1
        for member_idx in members:
            member_vec = self.data[member_idx]
            cosine_sim = self.calc_cosine(query_vec, member_vec)
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return ret

    def calc_cosine(self, vec_a, vec_b):
        return 1 - cosine(vec_a, vec_b)


def run_clustering(full_data, doc_id_file, filter_keyword_file, n_cluster, parent_direcotry, parent_description,\
                   cluster_keyword_file, hierarchy_file, doc_membership_file):
    dataset = SubDataSet(full_data, doc_id_file, filter_keyword_file)
    print('Start clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    
    if len(dataset.keywords) < n_cluster:
        print("Not enough nodes to cluster, skipping...")
        return None
    
    #Embedding - dictionary (token,embedding)
    clus = Clusterer(dataset.embeddings, n_cluster)
    clus.fit()
    print('Done clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    dataset.write_cluster_members(clus, cluster_keyword_file, parent_direcotry)
    center_names = dataset.write_cluster_centers(clus, parent_description, hierarchy_file)
    dataset.write_document_membership(clus, doc_membership_file, parent_direcotry)
    print('Done saving cluster results for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    return center_names

### Local Embedding methods

def read_files(folder, parent):
    print("[Local-embedding] Reading file:", parent)
    emb_file = '%s/embeddings.txt' % folder
    hier_file = '%s/hierarchy.txt' % folder
    keyword_file = '%s/keywords.txt' % folder ## here only consider those remaining keywords

    embs = load_embeddings(emb_file)
    keywords = set()
    cates = {}

    with open(keyword_file) as f:
        for line in f:
            keywords.add(line.strip('\r\n'))

    tmp_embs = {}
    for k in keywords:
        if k in embs:
            tmp_embs[k] = embs[k]
    embs = tmp_embs

    with open(hier_file) as f:
        for line in f:
            segs = line.strip('\r\n').split(' ')
            if segs[1] == parent: #('','ML')
                cates[segs[0]] = set()

    print('[Local-embedding] Finish reading embedding, hierarchy and keywords files.')

    return embs, keywords, cates

def relevant_phs(embs, cates, N):

    for cate in cates:
        worst = -100
        bestw = [-100] * (N + 1)
        bestp = [''] * (N + 1)
        cate_ph = cate

        for ph in embs:
            sim = cossim(embs[cate_ph], embs[ph])
            if sim > worst:
                for i in range(N):
                    if sim >= bestw[i]:
                        for j in range(N - 1, i - 1, -1):
                            bestw[j+1] = bestw[j]
                            bestp[j+1] = bestp[j]
                        bestw[i] = sim
                        bestp[i] = ph
                        worst = bestw[N-1]
                        break


        for ph in bestp[:N]:
            cates[cate].add(ph)

    print('Top similar phrases found.')

    return cates

def revevant_docs(text, reidx, cates):
    docs = {}
    idx = 0
    pd_map = {}
    for cate in cates:
        for ph in cates[cate]:
            pd_map[ph] = set()

    with open(text) as f:
        for line in f:
            docs[idx] = line
            idx += 1

    with open(reidx) as f:
        for line in f:
            segments = line.strip('\r\n').split('\t')
            doc_ids = segments[1].split(',')
            if len(doc_ids) > 0 and doc_ids[0] == '':
                continue
            pd_map[segments[0]] = set([int(x) for x in doc_ids])

    print('Relevant docs found.')

    return pd_map, docs


def run_word2vec(keywords, pd_map, docs, cates, folder):

    for cate in cates:

        c_docs = set()
        for ph in cates[cate]:
            c_docs = c_docs.union(pd_map[ph])

        print('Starting cell %s with %d docs.' % (cate, len(c_docs)))
        
        sub_folder = folder + cate + '/'
        input_f = sub_folder + 'text'
        output_f = sub_folder + 'embeddings.txt'
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        with open(input_f, 'w+') as g:
            for d in c_docs:
                g.write(docs[d])

        print('[Local-embedding] starting calling word2vec')
        print(input_f)
        print(output_f)
        
        sentences = LineSentence(input_f)
        
        model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=8,epochs=100)

        with open(output_f,'w') as f:
            f.write(str(len(model.wv.key_to_index.keys())) + ' ' + str(100) + '\n')
            for word in model.wv.key_to_index.keys():
                if word in keywords:
                    f.write(word + ' ')
                    for emb in model.wv[word]:
                        f.write(str(emb) + ' ')
                    f.write('\n')

        print('[Local-embedding] done training word2vec')


def main_local_embedding(folder, doc_file, reidx, parent, N):

    #N - n-expand
    #reidx - index_file
    #cates - Top N similar keyphrases to Parent node
    embs, keywords, cates = read_files(folder, parent) #Go to parent node
    cates = relevant_phs(embs, cates, int(N)) #Get relevant keyphrases based on n_expand parameter for each parent
    pd_map, docs = revevant_docs(doc_file, reidx, cates)
    run_word2vec(keywords, pd_map, docs, cates, folder)