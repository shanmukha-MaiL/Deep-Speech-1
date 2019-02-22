"""REFERENCES : 1)https://arxiv.org/abs/1412.5567 ----- DEEP_SPEECH_1
                2)https://github.com/chagge/DeepSpeech-1
"""
import os
import tensorflow as tf
from queue import PriorityQueue
from threading import Thread
import wave
from math import ceil
from glob import glob
import codecs
from Feature_inputs import wav_to_vector
from Text_preprocess import text_to_id,ctc_label_dense_to_sparse,validate_label
#import unicodedata
import fnmatch
from itertools import cycle

class DataSets(object):
    def __init__(self, train, validation, test):
        self._validation = validation
        self._test = test
        self._train = train

    def start_queue_threads(self, session):
        self._validation.start_queue_threads(session)
        self._test.start_queue_threads(session)
        self._train.start_queue_threads(session)

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test


class DataSet(object):
    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None, ])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None, ], []],
                                                  dtypes = [tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity = 2 * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self._example_queue.close(cancel_pending_enqueues=True)
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()


    def start_queue_threads(self, session, coord):
        self._coord = coord
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in range(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op)

    def _create_files_circular_list(self):
        priorityQueue = PriorityQueue()
        for txt_file in self._txt_files:
            wav_file = os.path.splitext(txt_file)[0] + ".wav"
            wav_file_size = os.path.getsize(wav_file)
            priorityQueue.put((wav_file_size, (txt_file, wav_file)))
        files_list = []
        while not priorityQueue.empty():
            priority, (txt_file, wav_file) = priorityQueue.get()
            files_list.append((txt_file, wav_file))
        return cycle(files_list)

    def _populate_batch_queue(self, session):
        for txt_file, wav_file in self._files_circular_list:
            if self._coord.should_stop():
                return
            source = wav_to_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                #target = unicodedata.normalize("NFKD", open_txt_file.read()).encode("ascii", "ignore")
                target = open_txt_file.read()
                target = text_to_id(target)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except tf.errors.CancelledError:
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) / float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, validation_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_validation=0, limit_test=0, limit_train=0, sets=[]):
    data_dir = os.path.join(data_dir, "LDC97S62")
    
    # Conditionally split wav data
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR1",
                     "audio-wavs-split/DR1")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR2",
                     "audio-wavs-split/DR2")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR3",
                     "audio-wavs-split/DR3")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR4",
                     "audio-wavs-split/DR4")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR5",
                     "audio-wavs-split/DR5")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR6",
                     "audio-wavs-split/DR6")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR7",
                     "audio-wavs-split/DR7")
    _maybe_split_wav(data_dir, "transcriptions", "audio-wavs/DR8",
                     "audio-wavs-split/DR8")

    _maybe_split_transcriptions(data_dir, "transcriptions")

    _maybe_split_sets(data_dir,["audio-wavs-split/DR1", "audio-wavs-split/DR2", "audio-wavs-split/DR3", "audio-wavs-split/DR4",
                      "audio-wavs-split/DR5","audio-wavs-split/DR6","audio-wavs-split/DR7","audio-wavs-split/DR8"],
                      "final_sets")

    # Create validation DataSet
    validation = None
    if "validation" in sets:
        validation = _read_data_set(data_dir, "final_sets/validation", thread_count, validation_batch_size, numcep,
                             numcontext, limit_validation)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = _read_data_set(data_dir, "final_sets/test", thread_count, test_batch_size, numcep,
                             numcontext, limit_test)

    # Create train DataSet
    train = None
    if "train" in sets:
        train = _read_data_set(data_dir, "final_sets/train", thread_count, train_batch_size, numcep,
                             numcontext, limit_train)

    # Return DataSets
    return DataSets(train, validation, test)

def _parse_transcriptions(trans_file):
    segments = []
    with open(trans_file, "r") as fin:
        for line in fin:
            if line.startswith("#")  or len(line) <= 1:
                continue

            start_time_beg = 0
            start_time_end = line.find(" ", start_time_beg)

            stop_time_beg = start_time_end + 1
            stop_time_end = line.find(" ", stop_time_beg)

            transcript_beg = stop_time_end + 1
            transcript_end = len(line)
            #print(line[transcript_beg:transcript_end].strip())
            
            if validate_label(line[transcript_beg:transcript_end].strip()) == None:
                continue
            #print('segment')
            segments.append({
                "start_time": float(line[start_time_beg:start_time_end]),
                "stop_time": float(line[stop_time_beg:stop_time_end]),
                "transcript": line[transcript_beg:transcript_end].strip().lower(),
            })
    #print(segments)
    return segments


def _maybe_split_wav(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    if os.path.exists(target_dir):
        print("Splitting wav files done !!!")
        return

    os.makedirs(target_dir)

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.TXT"):
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)
            #print(trans_file) 
            # Open wav corresponding to transcription file
            wav_filename = (os.path.splitext(os.path.basename(trans_file))[0]) + ".wav"
            wav_file = os.path.join(source_dir, wav_filename)

            print("splitting ",wav_file," according to ",trans_file)

            if not os.path.exists(wav_file):
                print("skipping. does not exist:" + wav_file)
                continue
            
            origAudio = wave.open(wav_file, "r")
            
            # Loop over segments and split wav_file for each segment
            for segment in segments:
                # Create wav segment filename
#                print('Segmenting !')
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                new_wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(
                    start_time) + "-" + str(stop_time) + ".wav"
                new_wav_file = os.path.join(target_dir, new_wav_filename)
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

            # Close origAudio
            origAudio.close()

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()


def _maybe_split_transcriptions(data_dir, original_data):
    source_dir = os.path.join(data_dir, original_data)
    wav_dirs = ["audio-wavs-split/DR1", "audio-wavs-split/DR2", "audio-wavs-split/DR3", "audio-wavs-split/DR4",
                "audio-wavs-split/DR5","audio-wavs-split/DR6","audio-wavs-split/DR7","audio-wavs-split/DR8"]

    if os.path.exists(os.path.join(source_dir, "split_transcriptions_done")):
        print("Done splitting transcriptions !!!!")
        return

    # Loop over transcription files and split them into individual files for
    # each utterance
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.TXT"):
#            
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                txt_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(
                    stop_time) + ".txt"
                wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(start_time) + "-" + str(
                    stop_time) + ".wav"

                transcript = validate_label(segment["transcript"])

                for wav_dir in wav_dirs:
                    if os.path.exists(os.path.join(data_dir, wav_dir, wav_filename)):
                        # If the transcript is valid and the txt segment filename does
                        # not exist create it
                        txt_file = os.path.join(data_dir, wav_dir, txt_filename)
                        if transcript != None and not os.path.exists(txt_file):
                            with open(txt_file, "w") as fout:
                                fout.write(transcript)
                        break

    with open(os.path.join(source_dir, "split_transcriptions_done"), "w") as fout:
        fout.write(
            "This file signals to the importer than the transcription of this source dir has already been completed.")


def _maybe_split_sets(data_dir, original_data, converted_data):
    target_dir = os.path.join(data_dir, converted_data)

    if os.path.exists(target_dir):
        return;

    os.makedirs(target_dir)

    filelist = []
    for dir in original_data:
        source_dir = os.path.join(data_dir, dir)
        filelist.extend(glob(os.path.join(source_dir, "*.txt")))

    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    validation_beg = int(0.8 * train_end)
    validation_end = train_end
    train_end = validation_beg

    test_beg = validation_end
    test_end = len(filelist)

    _maybe_split_dataset(filelist[train_beg:train_end], os.path.join(target_dir, "train"))
    _maybe_split_dataset(filelist[validation_beg:validation_end], os.path.join(target_dir, "validation"))
    _maybe_split_dataset(filelist[test_beg:test_end], os.path.join(target_dir, "test"))


def _maybe_split_dataset(filelist, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        for txt_file in filelist:
            new_txt_file = os.path.join(target_dir, os.path.basename(txt_file))
            os.rename(txt_file, new_txt_file)

            wav_file = os.path.splitext(txt_file)[0] + ".wav"
            new_wav_file = os.path.join(target_dir, os.path.basename(wav_file))
            os.rename(wav_file, new_wav_file)


def _read_data_set(work_dir, data_set, thread_count, batch_size, numcep, numcontext, limit=0):
    # Obtain list of txt files
    txt_files = glob(os.path.join(work_dir, data_set, "*.txt"))
    if limit > 0:
        txt_files = txt_files[:limit]

    # Return DataSet
    return DataSet(txt_files, thread_count, batch_size, numcep, numcontext)
