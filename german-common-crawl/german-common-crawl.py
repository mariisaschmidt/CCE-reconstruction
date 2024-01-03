# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""German Common Crawl"""

from __future__ import absolute_import, division, print_function
import datasets
import gzip
from ast import literal_eval


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{wenzek2020ccnet,
  title={CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data},
  author={Wenzek, Guillaume and Lachaux, Marie-Anne and Conneau, Alexis and Chaudhary, Vishrav and Guzm{\'a}n, Francisco and Joulin, Armand and Grave, {\'E}douard},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={4003--4012},
  year={2020}
}
"""

_DESCRIPTION = """\
German Only Extract from Common Crawl

This Dataset is for pretraining a German Language Model (Unsupervised) or tune a Multilingual Model specifically to German
"""

REPO_URL = "https://huggingface.co/datasets/mariisa/german-common-crawl/resolve/main/"

#TODO convert & upload all those files correctly
_URL_HEAD = [REPO_URL + file_name for file_name in [
    "de_head_0000_2015-48.txt.gz",
    "de_head_0000_2016-18.txt.gz",
    "de_head_0000_2016-44.txt.gz",
    "de_head_0000_2017-13.txt.gz",
    "de_head_0000_2017-30.txt.gz",
    "de_head_0000_2017-39.txt.gz",
    "de_head_0000_2017-51.txt.gz",
    "de_head_0000_2018-09.txt.gz",
    "de_head_0000_2018-17.txt.gz",
    "de_head_0000_2018-30.txt.gz",
    "de_head_0000_2018-39.txt.gz",
    "de_head_0000_2018-51.txt.gz",
    "de_head_0000_2019-18.txt.gz",
    "de_head_0000_2019-30.txt.gz",
    "de_head_0000_2019-47.txt.gz",
    "de_head_0000_2020-10.txt.gz",
    "de_head_0001_2016-44.txt.gz",
    "de_head_0001_2017-13.txt.gz",
    "de_head_0001_2017-30.txt.gz",
    "de_head_0001_2017-39.txt.gz",
    "de_head_0001_2017-51.txt.gz",
    "de_head_0001_2018-09.txt.gz",
    "de_head_0001_2018-17.txt.gz",
    "de_head_0001_2018-30.txt.gz",
    "de_head_0001_2018-39.txt.gz",
    "de_head_0001_2018-51.txt.gz",
    "de_head_0001_2019-09.txt.gz",
    "de_head_0001_2019-18.txt.gz",
    "de_head_0001_2019-30.txt.gz",
    "de_head_0001_2019-47.txt.gz",
    "de_head_0001_2020-10.txt.gz",
    "de_head_0002_2016-44.txt.gz",
    "de_head_0002_2017-13.txt.gz",
    "de_head_0002_2017-30.txt.gz",
    "de_head_0002_2017-39.txt.gz",
    "de_head_0002_2017-51.txt.gz",
    "de_head_0002_2018-09.txt.gz",
    "de_head_0002_2018-17.txt.gz",
    "de_head_0002_2018-30.txt.gz",
    "de_head_0002_2018-39.txt.gz",
    "de_head_0002_2018-51.txt.gz",
    "de_head_0002_2019-09.txt.gz",
    "de_head_0002_2019-18.txt.gz",
    "de_head_0002_2019-30.txt.gz",
    "de_head_0002_2019-47.txt.gz",
    "de_head_0002_2020-10.txt.gz",
    "de_head_0003_2016-44.txt.gz",
    "de_head_0003_2017-13.txt.gz",
    "de_head_0003_2017-30.txt.gz",
    "de_head_0003_2017-39.txt.gz",
    "de_head_0003_2017-51.txt.gz",
    "de_head_0003_2018-09.txt.gz",
    "de_head_0003_2018-17.txt.gz",
    "de_head_0003_2018-30.txt.gz",
    "de_head_0003_2018-39.txt.gz",
    "de_head_0003_2018-51.txt.gz",
    "de_head_0003_2019-09.txt.gz",
    "de_head_0003_2019-18.txt.gz",
    "de_head_0003_2019-30.txt.gz",
    "de_head_0003_2019-47.txt.gz",
    "de_head_0003_2020-10.txt.gz",
    "de_head_0004_2016-44.txt.gz",
    "de_head_0004_2017-30.txt.gz",
    "de_head_0004_2017-39.txt.gz",
    "de_head_0004_2017-51.txt.gz",
    "de_head_0004_2018-09.txt.gz",
    "de_head_0004_2018-17.txt.gz",
    "de_head_0004_2018-30.txt.gz",
    "de_head_0004_2018-39.txt.gz",
    "de_head_0004_2018-51.txt.gz",
    "de_head_0004_2019-09.txt.gz",
    "de_head_0004_2019-18.txt.gz",
    "de_head_0004_2019-30.txt.gz",
    "de_head_0004_2019-47.txt.gz",
    "de_head_0004_2020-10.txt.gz",
    "de_head_0005_2017-51.txt.gz",
    "de_head_0005_2018-09.txt.gz",
    "de_head_0005_2018-17.txt.gz",
    "de_head_0005_2018-30.txt.gz",
    "de_head_0005_2018-39.txt.gz",
    "de_head_0005_2018-51.txt.gz",
    "de_head_0005_2019-09.txt.gz",
    "de_head_0005_2019-18.txt.gz",
    "de_head_0005_2019-30.txt.gz",
    "de_head_0005_2019-47.txt.gz",
    "de_head_0005_2020-10.txt.gz",
    "de_head_0006_2018-09.txt.gz",
    "de_head_0006_2018-17.txt.gz",
    "de_head_0006_2018-30.txt.gz",
    "de_head_0006_2018-39.txt.gz",
    "de_head_0006_2018-51.txt.gz",
    "de_head_0006_2019-09.txt.gz",
    "de_head_0006_2019-18.txt.gz",
    "de_head_0006_2019-30.txt.gz",
    "de_head_0006_2019-47.txt.gz",
    "de_head_0006_2020-10.txt.gz",
    "de_head_0007_2018-30.txt.gz",
    "de_head_0007_2018-51.txt.gz",
    "de_head_0007_2019-09.txt.gz",
    "de_head_0007_2019-18.txt.gz",
    "de_head_0007_2019-47.txt.gz",
    "de_head_0007_2020-10.txt.gz",
]]


class GermanCommonCrawl(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="head", version=VERSION, description=""), #TODO fill description
    ]

    def _info(self):
        features = datasets.Features(
            {
                "url": datasets.Value("string"),
                "date_download": datasets.Value("string"),
                "digest": datasets.Value("string"),
                "length": datasets.Value("int32"),
                "nlines": datasets.Value("int32"),
                "source_domain": datasets.Value("string"),
                "title": datasets.Value("string"),
                "raw_content": datasets.Value("string"),
                "cc_segment": datasets.Value("string"),
                "original_nlines": datasets.Value("int32"),
                "original_length": datasets.Value("int32"),
                "language": datasets.Value("string"),
                "language_score": datasets.Value("int32"),
                "perplexity": datasets.Value("int32"),
                "bucket": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, txtget) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.name == "head":
            data_files = dl_manager.download(_URL_HEAD)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": data_files,
                },
            ),
        ]

    def _generate_examples(self, data_files):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        for filepath in data_files:
            with open(filepath, "rt", encoding="utf-8") as f:
#            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for id_, line in enumerate(f):
                    item = literal_eval(line)
                    yield id_, {
                        "url": item["url"],
                        "date_download": item["date_download"],
                        "digest": item["digest"],
                        "length": item["length"],
                        "nlines": item["nlines"],
                        "source_domain": item["source_domain"],
                        "title": item["title"],
                        "raw_content": item["raw_content"],
                        "cc_segment": item["cc_segment"],
                        "original_nlines": item["original_nlines"],
                        "original_length": item["original_length"],
                        "language": item["language"],
                        "language_score": item["language_score"],
                        "perplexity": item["perplexity"],
                        "bucket": item["bucket"],
                    }
