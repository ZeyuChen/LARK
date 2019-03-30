#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import paddle.fluid as fluid
import paddle_hub as hub

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from finetune.classifier import create_module, evaluate
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser

args = parser.parse_args()


def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                src_ids, sent_ids, pos_ids, input_mask, pooled_output, sequence_output = create_module(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config)

                exe = fluid.Executor(place)
                exe.run(startup_prog)

                init_pretraining_params(
                    exe,
                    args.init_pretraining_params,
                    main_program=startup_prog,
                    use_fp16=args.use_fp16)

                pooled_output_sign = hub.create_signature(
                    "pooled_output",
                    inputs=[src_ids, pos_ids, sent_ids, input_mask],
                    outputs=[pooled_output],
                    feed_names=["src_ids", "pos_ids", "sent_ids", "input_mask"],
                    fetch_names=["pooled_output"])

                sequence_output_sign = hub.create_signature(
                    "sequence_output",
                    inputs=[src_ids, pos_ids, sent_ids, input_mask],
                    outputs=[sequence_output],
                    feed_names=["src_ids", "pos_ids", "sent_ids", "input_mask"],
                    fetch_names=["sequence_output"])

                hub.create_module(
                    sign_arr=[pooled_output_sign, sequence_output_sign],
                    module_dir="./ernie_stable.hub_module",
                    module_info="./config/ernie_info.yml",
                    exe=exe,
                    assets=[args.vocab_path, args.ernie_config_path])


if __name__ == '__main__':
    print_arguments(args)

    main(args)
