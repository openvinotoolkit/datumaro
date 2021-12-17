#!/usr/bin/env python3

# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import subprocess

from packaging import version
import git
import toml

 # the initial version for the documentation site
MINIMUM_VERSION = version.Version('0.1.11')

def prepare_tags(repo):
    tags = {}
    for tag in repo.tags:
        tag_version = version.parse(tag.name)
        if tag_version >= MINIMUM_VERSION and not tag_version.is_prerelease:
            release_version = (tag_version.major, tag_version.minor)
            if release_version not in tags or tag_version > version.parse(tags[release_version].name):
                tags[release_version] = tag

    return tags.values()

def generate_versioning_config(filename, versions, url_prefix=''):
    def write_version_item(file_object, version, url):
        file_object.write('[[params.versions]]\n')
        file_object.write('version = "{}"\n'.format(version))
        file_object.write('url = "{}"\n\n'.format(url))

    with open(filename, 'w') as f:
        write_version_item(f, 'develop', '{}/'.format(url_prefix))
        for v in versions:
            write_version_item(f, v, '{}/{}'.format(url_prefix, v))

def git_checkout(tagname, repo_root):
    docs_dir = repo_root / 'site/content/en/docs'
    shutil.rmtree(docs_dir)
    repo.git.checkout(tagname, '--', 'site/content/en/docs')
    if version.Version(tagname) != MINIMUM_VERSION:
        images_dir = repo_root / 'site/content/en/images'
        shutil.rmtree(images_dir)
        repo.git.checkout(tagname, '--', 'site/content/en/images')

def change_version_menu_toml(filename, version):
    data = toml.load(filename)
    data['params']['version_menu'] = version

    with open(filename,'w') as f:
        toml.dump(data, f)

def generate_docs(repo, output_dir, tags):
    def run_hugo(content_loc, destination_dir):
        subprocess.run([ # nosec
                'hugo',
                '--destination',
                str(destination_dir),
                '--config',
                'config.toml,versioning.toml',
            ],
            cwd=content_loc,
        )

    repo_root = Path(repo.working_tree_dir)
    content_loc = repo_root / 'site'
    versioning_toml_path = content_loc / 'versioning.toml'

    generate_versioning_config(versioning_toml_path, (t.name for t in tags))
    change_version_menu_toml(versioning_toml_path, 'develop')
    run_hugo(content_loc, output_dir)

    generate_versioning_config(versioning_toml_path, (t.name for t in tags), '/..')
    for tag in tags:
        git_checkout(tag.name, repo_root)
        change_version_menu_toml(versioning_toml_path, tag.name)
        run_hugo(content_loc, output_dir / tag.name)

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    with git.Repo(repo_root) as repo:
        tags = prepare_tags(repo)
        generate_docs(repo, repo_root / 'public', tags)
