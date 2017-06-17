# -*- coding: utf-8 -*-
import pprint as pp
import sklearn_crfsuite
import numpy as np
from sklearn.metrics import f1_score

def load_data(filename):
	'''
	Загрузка данных
	:param filename: 
	:return: 
	'''
	with open(filename) as f:
		text = f.read()
		sents = text.split('\n\n')
		X = []
		y = []
		for sent in sents:
			sent_words = []
			sent_labels = []
			strings = sent.split('\n')
			for st in strings:
				el = st.split('\t')
				if el != ['']:
					word = el[0]
					label = el[1]
					sent_words.append(word)
					sent_labels.append(label)
			X.append(sent_words)
			y.append(sent_labels)
	return X, y


def word2features(word):
	'''
	Получение признаков, которые нужно извлечь для любого слова, вне зависимости от его положения в окне.
	Это такие признаки, как:
		1) токен в lower case;
		2) bias;
		3) токен - это цифра (или нет);
		4) первая и последняя буквы;
		5) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов.
	'''
	word = word.lower()
	word_features = [word, 1.0, word.isdigit(), word[0], word[-1]]  # признаки 1-4
	if len(word) > 1:  # префиксы и суффиксы в зависимости от длины слова
		word_features.extend([word[:2], word[-2:]])
	if len(word) > 2:
		word_features.extend([word[:3], word[-3:]])
	if len(word) > 3:
		word_features.extend([word[:4], word[-4:]])
	features = dict(zip(['word','bias', 'word_is_digit', 'pref[0]', 'suf[-1]',
						 'pref[:2]', 'suf[-2:]', 'pref[:3]', 'suf[-3:]', 'pref[:4]', 'suf[-4:]'], word_features))
	return features

def sent2features(sent):
	'''
	Все признаки для одного предложения.
	'''
	return [word2features(word) for word in sent]

print('Loading train set...')
train_set, y_train = load_data('opencorpora_train.txt')
print('Train set is loaded. Train set to features...')
X_train = [sent2features(sent) for sent in train_set]
print('X_train is formed.')
print('Loading test set...')
test_set, y_test = load_data('opencorpora_test.txt')
print('Test set is loaded. Test set to features...')
X_test = [sent2features(sent) for sent in test_set]
print('X_test is formed.')

#Классификатор, обучение и прогнозирование.

print('Training starts')
crf = sklearn_crfsuite.CRF(c1=0.01, c2=0.01, all_possible_transitions=True, max_iterations=150)
crf.fit(X_train, y_train)
print('Prediction starts')
y_pred = crf.predict(X_test)

y_test_new = []
y_pred_new = []
for el in y_test:
	y_test_new.extend(el)
for el in y_pred:
	y_pred_new.extend(el)
y_test_1d = np.array(y_test_new)
y_pred_1d = np.array(y_pred_new)

print('F1-macro: ', f1_score(y_test_1d, y_pred_1d, average='macro'))
print('F1-micro: ', f1_score(y_test_1d, y_pred_1d, average='micro'))
