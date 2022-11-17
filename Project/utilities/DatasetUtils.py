from IPython import display
from typing import Optional

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf


_SAMPLING_RATE = 16000

def create_sequences(
		dataset: tf.data.Dataset, 
		keys: list[str],
		seq_length: int,
		vocab_size: int = 128
	) -> tf.data.Dataset:
	"""Returns TF Dataset of sequence and label examples."""
	seq_length = seq_length + 1

	# Take 1 extra for the labels
	windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

	# `flat_map` flattens the" dataset of datasets" into a dataset of tensors
	flatten = lambda x: x.batch(seq_length, drop_remainder=True)
	sequences = windows.flat_map(flatten)

	# Normalize note pitch
	def scale_pitch(x):
		x = x/[vocab_size, 1.0, 1.0, 1.0, 1.0]
		return x

	# Split the labels
	def split_labels(sequences):
		inputs = sequences[:-1]
		labels_dense = sequences[-1]
		labels = {key:labels_dense[i] for i,key in enumerate(keys)}

		return inputs, labels

	return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def display_audio(pm: pretty_midi.PrettyMIDI, seconds=1000):
	waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
	# Take a sample of the generated waveform to mitigate kernel resets
	waveform_short = waveform[:seconds*_SAMPLING_RATE]
	return display.Audio(waveform_short, rate=_SAMPLING_RATE)


def midi_to_notes(midi_file: str) -> pd.DataFrame:
	pm = pretty_midi.PrettyMIDI(midi_file)
	instrument = pm.instruments[0]
	notes = collections.defaultdict(list)

	# Sort the notes by start time
	sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
	prev_start = sorted_notes[0].start

	for note in sorted_notes:
		start = note.start
		end = note.end
		notes['pitch'].append(note.pitch)
		notes['start'].append(start)
		notes['end'].append(end)
		notes['step'].append(start - prev_start)
		notes['duration'].append(end - start)
		prev_start = start
	
	return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def notes_to_midi(
		notes: pd.DataFrame,
		out_file: str, 
		instrument_name: str,
		velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:

	pm = pretty_midi.PrettyMIDI()
	instrument = pretty_midi.Instrument(
		program=pretty_midi.instrument_name_to_program(instrument_name)
	)
	for i, note in notes.iterrows():
		note = pretty_midi.Note(
			velocity=velocity,
			pitch=int(note['pitch']),
			start=float(note['start']),
			end=float(note['end']),
		)
		instrument.notes.append(note)

	pm.instruments.append(instrument)
	pm.write(out_file)
	return pm


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
	if count:
		title = f'First {count} notes'
	else:
		title = f'Whole track'
		count = len(notes['pitch'])
	plt.figure(figsize=(20, 4))
	plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
	plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
	plt.plot(
		plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker="."
	)
	plt.xlabel('Time [s]')
	plt.ylabel('Pitch')
	_ = plt.title(title)
	return plt.show()


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
	plt.figure(figsize=[15, 5])
	plt.subplot(1, 3, 1)
	sns.histplot(notes, x="pitch", bins=20)

	plt.subplot(1, 3, 2)
	max_step = np.percentile(notes['step'], 100 - drop_percentile)
	sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

	plt.subplot(1, 3, 3)
	max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
	sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))
	return plt.show()


def rmse(y, y_pred):
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y - y_pred)))