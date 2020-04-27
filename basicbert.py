# basicbert.py

# Written by David Stein (david@djstein.com).
# See https://www.djstein.com/basicbert/ for more info.
# Source: https://github.com/neuron-whisperer/basicbert

# This code is a wrapper class for the Google BERT transformer model:
#    https://github.com/google-research/bert

import collections, csv, ctypes, datetime, logging, modeling, numpy
import os, random, shutil, sys, tensorflow as tf, time, tokenization
from tensorflow.contrib import predictor
from tensorflow.python.util import deprecation

# these settings are positioned here to suppress warnings from run_classifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from run_classifier import *

# ====== Main Class ======

class BERT(logging.Filter):

	def __init__(self, config = {}):
	
		# read values from config file, or choose defaults
		self.script_dir = os.path.dirname(os.path.realpath(__file__))
		if os.path.isfile(os.path.join(self.script_dir, 'config.txt')):
			with open(os.path.join(self.script_dir, 'config.txt'), 'rt') as file:
				for line in (line.strip() for line in file.readlines()):
					if len(line) == 0 or line[0] == '#' or line.find('=') == -1:
						continue
					params = list(p.strip() for p in line.split('='))
					if len(params) == 2 and params[0] not in config:
						config[params[0]] = params[1]
		self.data_dir = os.path.join(self.script_dir, config.get('data_dir', 'input/'))
		self.output_dir = os.path.join(self.script_dir, config.get('output_dir', 'output/'))
		self.bert_config_file = os.path.join(self.script_dir, config.get('bert_config_file', 'bert_base/bert_config.json'))
		self.vocab_file = os.path.join(self.script_dir, config.get('vocab_file', 'bert_base/vocab.txt'))
		self.labels_file = config.get('labels_file', os.path.join(self.data_dir, 'labels.txt'))
		self.init_checkpoint = os.path.join(self.script_dir, config.get('init_checkpoint', 'bert_base/bert_model.ckpt'))
		self.exported_model_dir = config.get('exported_model_dir', '')
		self.tf_output_file = config.get('tf_output_file', None)
		self.tf_output_file = os.path.join(self.script_dir, self.tf_output_file) if self.tf_output_file else None
		self.do_lower_case = True if config.get('train_batch_size', False).lower() == 'true' else False
		self.train_batch_size = int(config.get('train_batch_size', 25))
		self.num_train_epochs = int(config.get('num_train_epochs', 100))
		self.warmup_proportion = float(config.get('warmup_proportion', 0.05))
		self.learning_rate = float(config.get('learning_rate', 5e-5))
		self.max_seq_length = int(config.get('max_seq_length', 256))
		self.save_checkpoint_steps = int(config.get('save_checkpoint_steps', 10000))

		# erase TensorFlow log in output
		if self.tf_output_file:
			with open(self.tf_output_file, 'wt') as log:
				log.write(f'{datetime.datetime.now():%Y%m%d %H:%M:%S %p}: Starting BERT\n')
		
		# turn off warnings
		logger = logging.getLogger('tensorflow')
		logger.setLevel(logging.INFO)
		logger.addFilter(self)

		# assorted configuration
		self.examples = None; self.loaded_model = None
		self.epoch = 0; self.num_train_steps = None; self.loss = None
		csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
		tokenization.validate_case_matches_checkpoint(self.do_lower_case, self.init_checkpoint)
		# tf.io.gfile.makedirs(self.output_dir)
		self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
		self.labels = self._get_labels()
		self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
		self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
		self.run_config = tf.contrib.tpu.RunConfig(cluster=None, master=None, model_dir=self.output_dir, save_checkpoints_steps=True)

	def filter(self, record):
		if self.tf_output_file:
			with open(self.tf_output_file, 'at') as log:		# log all output
				log.write(f'{datetime.datetime.now():%Y%m%d %H:%M:%S %p}: {record.getMessage()}\n')
		if record.msg.find('Saving checkpoints for') > -1:
			step = int(record.args[0])
			now = datetime.datetime.now()
			print(f'\r{now:%Y%m%d %H:%M:%S %p}: Epoch {self.epoch + 1} Step {(step % self.steps_per_epoch) + 1} / {self.steps_per_epoch}    ', end='')
		elif record.msg.find('Loss for final step') > -1:
			self.loss = float(record.args[0])
		return False

	def _get_labels(self):			# get labels from labels file or training file
		if self.labels_file and os.path.isfile(self.labels_file):
			with open(self.labels_file, 'rt') as file:
				return list(line.strip() for line in file.readlines() if len(line.strip()) > 0)
		lines = DataProcessor._read_tsv(os.path.join(self.data_dir, 'train.tsv'))
		return list(line[1].strip() for line in lines if len(line) >= 4 and len(line[1].strip()) > 0)
	
	def _get_ids(self, filename):		# get identifiers from first column of TSV
		labels = []
		with open(filename, 'rt') as file:
			lines = file.readlines()
		lines = list(line.strip().split('\t') for line in lines if len(line.strip()) > 0)
		return list(line[0].strip() for line in lines if len(line[0].strip()) > 0)

	def _create_estimator(self):
		num_warmup_steps = 0 if self.num_train_steps is None else int(self.num_train_steps * self.warmup_proportion)
		self.model_fn = model_fn_builder(bert_config=self.bert_config, num_labels=len(self.labels),
			init_checkpoint=self.init_checkpoint, learning_rate=self.learning_rate,
			num_train_steps=self.num_train_steps, num_warmup_steps=num_warmup_steps, use_tpu=False,
			use_one_hot_embeddings=False)
		self.estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False, model_fn=self.model_fn,
			config=self.run_config, train_batch_size=self.train_batch_size,
			eval_batch_size=1, predict_batch_size=1)

	def _create_examples(self, lines, set_type):
		self.examples = []
		for (i, line) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			if set_type == "test":
				if i == 0:		# discard header row
					continue
				text_a = tokenization.convert_to_unicode(line[1])
				label = self.labels[0]
			else:
				text_a = tokenization.convert_to_unicode(line[3])
				label = tokenization.convert_to_unicode(line[1])
			self.examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
	
	def _prepare_input_fn(self, mode):
		tsv = DataProcessor._read_tsv(os.path.join(self.data_dir, mode + '.tsv'))
		self._create_examples(tsv, mode)
		self.steps_per_epoch = int(len(self.examples) / self.train_batch_size)
		self.num_train_steps = self.steps_per_epoch * self.num_train_epochs
		self._create_estimator()
		record = os.path.join(self.output_dir, mode + '.tf_record')
		file_based_convert_examples_to_features(self.examples, self.labels, self.max_seq_length,
			self.tokenizer, record)
		return file_based_input_fn_builder(input_file=record, seq_length=self.max_seq_length,
			is_training=(mode == 'train'), drop_remainder=(mode == 'train'))
	
	def _load_model(self):
		if self.loaded_model is not None:
			return
		filename = self.find_exported_model()
		if filename == '':
			print('Error: No exported model specified or located.'); return
		self.loaded_model = predictor.from_saved_model(filename)

# ====== Callable Utility Functions ======
	
	def prepare_data(self, train, dev, input_filename = None, output_path = None):
		""" Prepares training data file based on train and dev ratios. """
		input_filename = input_filename or os.path.join(self.script_dir, 'data.csv')
		output_path = output_path or self.data_dir
		records = []; t = '\t'; n = '\n'
		with open(input_filename, 'rt') as file:
			for line in list(line.strip() for line in file.readlines() if len(line.strip()) > 0):
				params = line.split(',')
				if len(params) >= 3:
					records.append((params[0].strip(), params[1].strip(), ','.join(params[2:]).strip()))
		random.shuffle(records)
		train_index = int(len(records) * train); dev_index = int(len(records) * (train + dev))
		if train_index > 0:
			with open(os.path.join(output_path, 'train.tsv'), 'wt') as file:
				for record in records[:train_index]:
					file.write(f'{record[0]}{t}{record[1]}{t}a{t}{record[2]}{n}')
		if dev_index > train_index:
			with open(os.path.join(output_path, 'dev.tsv'), 'wt') as file:
				for record in records[train_index:dev_index]:
					file.write(f'{record[0]}{t}{record[1]}{t}a{t}{record[2]}{n}')
		if dev_index < len(records):
			with open(os.path.join(output_path, 'test.tsv'), 'wt') as file:
				with open(os.path.join(output_path, 'test-labels.tsv'), 'wt') as labels_file:
					for record in records[dev_index:]:
						file.write(f'{record[0]}{t}{record[2]}{n}')		# write identifier and text
						labels_file.write(f'{record[0]}{t}{record[1]}{n}')		# write identifier and label
		self.export_labels(os.path.join(output_path, 'labels.txt'))

	def find_exported_model(self):
		""" Finds the latest exported model based on timestamps in output folder. """
		best_path = self.exported_model_dir
		if best_path and os.path.isfile(os.path.join(best_path, 'saved_model.pb')):
			return best_path
		best_path = ''; best = None
		files = os.listdir(self.output_dir)
		for dir in files:
			path = os.path.join(self.output_dir, dir)
			if os.path.isdir(path) and dir.isnumeric():
				if not best or int(dir) > best:
					if os.path.isfile(os.path.join(path, 'saved_model.pb')):
						best = int(dir); best_path = path
		return best_path

	def export_labels(self, filename = None):
		""" Exports the label set to a file. One label per line. """
		filename = filename or self.labels_file
		with open(filename, 'wt') as f:
			for label in self.labels:
				f.write(f'{label}\n')
	
	def reset(self, output = False):
		""" Resets the training state of the model. """
		for file in os.listdir(self.output_dir):
			if os.path.isfile(os.path.join(self.output_dir, file)):
				os.unlink(os.path.join(self.output_dir, file))
			else:
				shutil.rmtree(os.path.join(self.output_dir, file), ignore_errors = True)
		if output:
			print('Reset input.')
	
# ====== Callable Primary Functions ======
	
	def train(self, num_epochs = None, output = False):
		""" Trains the model for a number of epochs."""
		fn = self._prepare_input_fn('train')
		epochs = num_epochs or self.num_train_epochs
		while self.epoch < epochs:
			steps = (self.epoch + 1) * self.steps_per_epoch
			epoch_start = time.time()
			self.estimator.train(input_fn=fn, max_steps=steps)
			duration = time.time() - epoch_start
			if self.loss is None:			# epoch was skipped
				if num_epochs:					# increment so that we run at least (num_epochs)
					epochs += 1
			elif output:
				print(f'Done. Loss: {self.loss:0.4f}. Duration: {int(duration)} seconds.')
			self.epoch +=1
		self.export_labels()
		return self.loss

	def eval(self, output = False):
		""" Evaluates the contents of dev.tsv and prints results. """
		fn = self._prepare_input_fn('dev')
		results = self.estimator.evaluate(input_fn=fn)
		output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
		if output:
			print('Evaluation results:')
			for key in sorted(results.keys()):
				print(f'	{key} = {str(results[key])}')
		return results

	def test(self, output = False):
		""" Tests the contents of test.tsv and prints results. """
		ids = self._get_ids(os.path.join(self.data_dir, 'test.tsv'))
		# get labels from test input
		fn = self._prepare_input_fn('test')
		records = self.estimator.predict(input_fn=fn)
		output_predict_file = os.path.join(self.output_dir, "test_results.tsv")
		results = []
		if output:
			print("Prediction results:")
		for (i, prediction) in enumerate(records):
			probabilities = prediction["probabilities"]
			probabilities_dict = {}
			for j in range(len(probabilities)):
				probabilities_dict[self.labels[j]] = probabilities[j]
			best_class = int(numpy.argmax(probabilities))
			if output:
				print(f'Input {i+1} ({ids[i]}): {self.labels[best_class]} ({probabilities[best_class] * 100.0:0.2f}%)')
			results.append((ids[i], self.labels[best_class], probabilities[best_class], probabilities_dict))
		return results

	def export(self, path = None, output = False):
		""" Exports the model to output_dir or to the specified path. """
		self._create_estimator()
		def serving_input_fn():
			label_ids = tf.placeholder(tf.int32, [None], name='label_ids')	
			input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_ids')
			input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_mask')
			segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name='segment_ids')
			input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
				'label_ids': label_ids,
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids,
			})()
			return input_fn
		self.estimator._export_to_tpu = False
		model_dir = path or self.output_dir 
		self.estimator.export_saved_model(model_dir, serving_input_fn)
		return self.find_exported_model()

	def predict(self, input, output = False):
		""" Predicts the classification of an input string. """
		self._load_model()
		text_a = tokenization.convert_to_unicode(input)
		example = InputExample(guid='0', text_a=text_a, text_b=None, label=self.labels[0])
		feature = convert_single_example(0, example, self.labels, self.max_seq_length, self.tokenizer)		
		result = self.loaded_model({'input_ids': [feature.input_ids], 'input_mask': [feature.input_mask], 'segment_ids': [feature.segment_ids], 'label_ids': [feature.label_id]})
		probabilities = result['probabilities'][0]
		all_predictions = {}
		for i, probability in enumerate(probabilities):
			all_predictions[self.labels[i]] = probability
		best_class = int(numpy.argmax(probabilities))
		if output:
			print(f'Prediction: {self.labels[best_class]} ({probabilities[best_class]})')
			print(f'  All predictions: {all_predictions}')
		return((self.labels[best_class], probabilities[best_class], all_predictions))

# ====== Main Function ======

if __name__ == "__main__":
	try:
		command = sys.argv[1].lower() if len(sys.argv) >= 2 else ''
		functions = ['train', 'eval', 'test', 'export', 'predict', 'prepare_data', 'reset']
		if len(command) == 0 or command not in functions:
			print(f'syntax: bert.py ({" | ".join(functions)})'); sys.exit(1)
		b = BERT()
		if command == 'predict':
			input = ' '.join(sys.argv[2:])
			b.predict(input, True)
		elif command == 'train' and len(sys.argv) > 2 and sys.argv[2].isnumeric():
			b.train(int(sys.argv[2]), True)
		elif command == 'export':
			filename = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
			b.export(filename, True)
		elif command == 'prepare_data':
			train = 0.95 if len(sys.argv) < 3 else float(sys.argv[2])
			dev = 0.025 if len(sys.argv) < 4 else float(sys.argv[3])
			input_filename = None if len(sys.argv) < 5 else sys.argv[4]
			output_path = None if len(sys.argv) < 6 else sys.argv[5]
			b.prepare_data(train, dev, input_filename, output_path)
		else:
			getattr(b, sys.argv[1].lower())(True)
	except KeyboardInterrupt:			# gracefully ctrl-c interrupts
		pass
