# basicbert

A wrapper class and usage guide for Google's Bidirectional Encoder Representation from Transformers (BERT) text classifier.

Written by David Stein (david@djstein.com). Also available at [https://www.djstein.com/basicbert](https://www.djstein.com/basicbert).

## The Short Version

The purpose of this project is to provide a wrapper class for the Google BERT transformer-based machine learning model and a usage guide for text classification. The objective is to enable developers to apply BERT out-of-the-box for ordinary text classification tasks.

## Background

Transformers have become the primary machine learning technology for text processing tasks. One of the best-known transformer platforms is [the Google BERT model](https://github.com/google-research/bert), which features several different pretrained models that may be generally applied to a variety of tasks with a modest amount of training. The BERT codebase includes a basic file (`run_classifier.py`) that can be configured for different tasks via a set of command-line parameters.

Despite the impressive capabilities of Google BERT, the codebase suffers from a variety of limitations and disadvantages, such as the following:

* BERT is based on TensorFlow, and therefore suffers from the TensorFlow 1.x / 2.x dichotomy. The original BERT codebase (linked above) is TensorFlow 1.x code, some of which will not run natively in a TensorFlow 2.x environment. Efforts are under way to [translate BERT into TensorFlow 2.x](https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22), but this has created a deep divide in the available code for various BERT applications and discussion topics.

* BERT exhibits the standard TensorFlow problem of generating *a vast* amount of output, which commingles informational notices, progress indicators, and warnings, including "deprecated code" messages. It is not easy to turn off the excessive output or to filter out the parts that are relevant to the task at hand. Additionally, the warnings provide suggestions for migrate to TensorFlow 2.x, and some of them are not actually applicable (due to unwritten portions of the tensorflow.compat.v1 codebase!)

* `run_classifier.py` provides an abstract DataProcessor class, and then requires users to choose among several different subclasses for different examples: ColaProcessor, MnliProcessor, MrpcProcessor, and XnliProcessor. The README does not explain what these processors are. The codebase merely indicates, unhelpfully, that these processors are used for the CoLA, MNLI, MRPC, and XNLI data sets. Nothing in the repository guides users in choosing among the provided DataProcessors or writing their own in order to use BERT for their own data sets or applications.

* BERT is written to use several of Google's machine learning initiatives: training on GPUs or TPUs, hosting models on [TensorFlow Hub](https://www.tensorflow.org/hub), and [hosting trained BERT models to serve clients from the cloud](https://bert-as-service.readthedocs.io/en/latest/). Unfortunately, these features are not supplemental to a vanilla BERT implementation that performs basic text classification. Rather, the BERT codebase expects to use these features by default, and then requires developers to figure out how to disable them. For example, BERT *requires* the use of the TPUEstimator class, and the TPU-based features must be turned off to force BERT into a CPU-training context. Also, BERT features parameters that are only used for distributed TPU-based training (such as `eval_batch_size`, `predict_batch_size`, `iterations_per_loop`) and that do not even make sense in other contexts - but the BERT codebase does not clearly explain these features.

* The BERT codebase is poorly written and unnecessarily complicated. For example:

	* Configuration is only by way of a long string of command-line parameters.
	
	* The standard example code (`run_classifier.py`) requires input files to be formatted with specific names ("train.tsv", "dev.tsv", and "test.tsv"). Also, the established format is peculiar: the train and dev sets require four columns including *a completely useless* third column; and test.tsv requires a header row that is silently discarded ignored (the others do not).
	
	* BERT lacks some standard features, such as displaying a per-epoch loss in the manner that we have come to expect from Keras training.

	* BERT does not save the labels as part of the model, so this basic information must be persisted somewhere by the user.
	
	* <a name="timestamp">BERT can export a trained model to a named path, but it insists on creating a subfolder that is arbitrarily named according to a timestamp - such that loading the model *that was just exported* requires [clumsily searching the contents of the output folder](https://guillaumegenthial.github.io/serving-tensorflow-estimator.html#the-problem).</a>

These and many other problems arose during my initial experimentation with BERT for a simple project. The entire codebase and documentation entirely fail to answer basic questions, like: How do I export a trained model, or use one to predict the class of an input on the fly, in the manner of an API?

My initial work with BERT required a significant amount of time examining and experimenting with the codebase to understand and circumvent these problems, and to wrangle BERT into a form that can be used with a minimum of hassle. The result is a simple wrapper class that can be (a) configured via a simple text configuration file and (b) invoked with simple commands to perform everyday classification tasks.

## Implementation

The heart of this project is [`basicbert.py`](https://www.github.com/neuron-whisperer/basicbert/blob/master/basicbert.py), which is designed to run in a Python 3 / TensorFlow 1.15.0 environment.

`basicbert.py` has been adapted from the Processor subclasses of `run_classifier.py`, and it reuses as much of the base code as possible. The wrapper exposes a few simple functions: `reset()`, `train()`, `eval()`, `test()`, `export()`, and `predict()`. It can be used in this manner to perform text classification of .tsv files with an arbitrarily collected set of labels. A set of utility functions is also provided to prepare the training data and to reset the training state.

`basicbert.py` can be configured by creating or editing a file called `config.txt` in the same folder as `basicbert.py`. The configuration file has a simple key/value syntax (e.g.: `num_train_epochs = 10`). If the file does not exist or does not contain some options, `basicbert.py` will default to some standard values.

`basicbert.py` subclasses the `logging.Filter` class and hooks a filter function to the TensorFlow logging process, which redirects all TensorFlow output to `filter(self, record)`. The filter function scrapes a minimal amount of needed data (training progress and loss) from the voluminous TensorFlow output and discards the rest. For debugging, `basicbert.py` can be configured to save the complete TensorFlow output to a separate text file (by setting the `tf_output_file` configuration parameter).

`basicbert.py` can export the model from the latest checkpoint and load it to perform inference. This likely requires saving the labels used for training, which BERT does not do by default. `basicbert.py` saves the list as `labels.txt` in the input folder, but this is configurable via `config.txt`.

## Usage

The following steps will train a BERT model and perform some testing and prediction.

### Step 1: Prepare Codebase

* Create a base folder.

* Install TensorFlow 1.15 (preferably, but not necessarily, within a virtual environment within the base folder):

		python3 -m venv basicbert-env
		source basicbert-env/bin/activate
		pip3 install tensorflow==1.15

* Download [the Google BERT master repository from GitHub](https://github.com/google-research/bert). Extract it and move all of the files into the base folder.

* Download one of the Google BERT pretrained models from GitHub (such as [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)). Make a subfolder in the base folder called `bert_base` and extract the model files there. (The files should be stored in the `bert_base` folder, not `bert_base/bert_base_uncased/`, *etc.*)

* Download [`basicbert.py`](https://www.github.com/neuron-whisperer/basicbert/blob/master/basicbert.py) and [`config.txt`](https://www.github.com/neuron-whisperer/basicbert/blob/master/config.txt) from this repository and copy them to the base folder.

* Do one of the following two options:

  * Download [`run_classifier.py`](https://www.github.com/neuron-whisperer/basicbert/blob/master/run_classifier.py) from this repository and copy it to the base folder, overwriting `run_classifier.py` from the Google BERT master repository.
  
  * Edit `run_classifier.py` from the Google BERT master repository and insert the following line of code (at line 681 of the current version of `run_classifier.py`, but this could change):
  
		training_hooks=[tf.train.LoggingTensorHook({'loss': total_loss}, every_n_iter=1)],

...as follows:

		output_spec = tf.contrib.tpu.TPUEstimatorSpec(
			mode=mode,
			loss=total_loss,
			train_op=train_op,
			training_hooks=[tf.train.LoggingTensorHook({'loss': total_loss}, every_n_iter=1)],
			scaffold_fn=scaffold_fn)

(Why is this necessary? Because BERT calculates the loss during training, but only reports the per-epoch loss during training if you request it - and `run_classifier.py` does not. See [this GitHub thread](https://github.com/google-research/bert/issues/70) for more information about this modification.)

### Step 2: Prepare Data

* Make a subfolder in the base folder called `input` in the base folder.

* Prepare the TSV files using one of the following three options:

	* Generate `train.tsv`, `dev.tsv`, and `test.tsv`, for example, as discussed [here](https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7). Yes, the formats are peculiar, including a completely useless column for no particular reason. Save the files in the input directory. **Note:** `basicbert.py` allows you to use any labels you want for your sentences. The only cautionary note is that *all* labels that are present in your evaluation data *must* be included in at least one training data row.
	
	* Prepare a master input file as a three-column CSV file, save it in the same folder as `basicbert.py`, and use `prepare_data()` to generate the tsv ([see below](#prepare_data)).
	
	* Download `train.tsv`, `dev.tsv`, and `test.tsv` from any source you like. If you would like to experiment with an example data set, download [this example training data set](https://www.github.com/neuron-whisperer/basicbert/blob/master/example_tsvs.zip) from the basicbert GitHub repository.

### Step 3: Use basicbert

* Review `config.txt` and make any changes that you'd like to the configuration.

* Train the BERT model using the following terminal command:

		python3 basicbert.py train

By default, `basicbert.py` will train a BERT model on ten epochs of the test data, reporting the loss for each epoch and saving checkpoints along the way. The training process can be canceled at any point, and will automatically resume from the last checkpoint.

If `basicbert.py` finishes training for the number of epochs indicated in `config.txt`, then subsequent training commands will be disregarded unless the number of epochs is increased. Alternatively, you may specify the number of training epochs, which will be completed irrespective of the number of previously completed epochs:

		python3 basicbert.py train 3

`basicbert` can also be used programmatically:

		from basicbert import *
		bert = BERT()
		bert.train()	 # returns loss for the last training epoch

The BERT() initializer attempts to load its configuration from `config.txt` in the same folder as `basicbert.py`. If `config.txt` is not present, BERT will use predefined defaults. The BERT initializer optionally accepts a configuration dictionary and uses any values in the dictionary will take highest priority, and will fall back on `config.txt` or defaults for any values missing from the dictionary. 

* Run the BERT model in evaluation mode (via terminal or Python) using either of the following:

		python3 basicbert.py eval
		bert.eval()

`eval()` returns a dictionary of results, with keys: `eval_accuracy, eval_loss, global_step, loss`.

* Run the BERT model in test mode using either of the following:

		python3 basicbert.py test
		bert.test()

`test()` returns an array of tuples, each representing the test result for one example. Each tuple has the following format: `(sample_id, best_label, best_confidence, {each_label: each_confidence})`.

* Export the BERT model:

		python3 basicbert.py export
		bert.export()

As [previously noted](#timestamp), BERT is configured by default to export models to a subfolder of the output folder, where the subfolder is named by a timestamp. You may move the files to any other path you choose, and may indicate the new location in `config.txt`. If you choose to leave them in the output folder, when `basicbert.py` loads the model during prediction, it will examine the subfolders and choose the subfolder with the largest number (presumably the last and best checkpoint). `export()` returns the path of the exported model.

* Use an exported BERT model for inference:

		python3 basicbert.py predict (input sentence)
		bert.predict(text)

Example:

		python3 basicbert.py predict The quick brown fox jumped over the lazy dogs.
		bert.predict('The quick brown fox jumped over the lazy dogs.')

The command-line call displays the predicted class, the probability, and the complete list of classes and probabilities. `predict()` returns a tuple of (string predicted_class, float probability, {string class: float probability}).

**Note:** As previously noted, an exported BERT model does not include the label set. Without the labels, BERT will have no idea how to map the predicted categories to the assigned labels. To address this deficiency, `predict()` looks for either `labels.txt` or `train.tsv` to retrieve the label set. A path to the label set file can be specified in `config.txt`.

## Utility Functions

The following utility functions are also available for the following tasks:

* <a name="prepare_data">Prepare .tsv data sets from a master data set:</a>

		python3 basicbert.py prepare_data 0.95 0.025
		bert.prepare_data(0.95, 0.025, input_filename, output_filename)

`prepare_data()` prepares .tsv files for use with BERT. It reads the specified file (or, by default, `data.csv` in the script folder), which should be a CSV that is formatted as follows:

		unique_per_sample_identifier, label, text

For example:

		sentence_001, label_1, This is a first sentence to be classified.

		sentence_002, label_2, This is a second sentence to be classified.

Rows are separated by newline characters. Sentences may contain or omit quote marks. Sentences may contain commas (even without quote marks).

The function accepts two floating-point parameters: train and dev, each indicating the number of rows to store in each file. The number of samples for the test set is calculated as (1.0 - train - dev). The function reads the sample data, shuffles the rows, and determines the number of samples to store in each file. It then writes the following files to the same folder:

`train.tsv`: tab-separated file for training data set

`dev.tsv`: tab-separated file for validation data set

`test.tsv`: tab-separated file for test data set

`labels.txt`: newline-separated list of labels

`test-labels.tsv`: tab-separated file of correct labels for test data, formatted as follows:

		unique_per_sample_identifier \t label

* Find an exported model in the output_dir folder and return its path:

		bert.find_exported_model()

* Export the labels from the training data set (optionally specifying the output filename):

		bert.export_labels()

* Reset the training of the BERT model (deletes the contents of the output folder):

		python3 basicbert.py reset
		bert.reset()

---
