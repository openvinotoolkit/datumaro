## Basic manual for website editing

### Edit or add documentation pages

To edit and/or add documentation, you need to have a [GitHub](https://github.com/login) account.
To change documentation files or add a documentation page,
simply click `Edit this page` on the page you would like to edit.
If you need to add a child page, click `Create child page`.

If you need to edit the text that has the markup [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet),
click on the `Fork this repository` button.

Read how to edit files for github ([GitHub docs](https://docs.github.com/en/github/managing-files-in-a-repository/editing-files-in-another-users-repository)).

Please note that files have a markup for correct display on the site:
the title, the title of the link, the weight (affects the order of files
display on the sidebar), the description and the tags:

    ---
    title: 'Title'
    linkTitle: 'Link Title'
    weight: 1
    description: >
        Description
    tags: [ 'Tag1', 'Tag2' ]
    ---

### Start site localy

To start the site locally, you need a recent [extended version hugo](https://github.com/gohugoio/hugo/releases)
(recommend version 0.75.0 or later).
Open the most recent release and scroll down until you find
a list of Extended versions. [Read more](https://gohugo.io/getting-started/installing/#quick-install)

Add a path to "hugo" in the "Path" environment variable.

Clone a repository branch containing the site. For example,
using a git command:

    git clone --branch <branchname> <remote-repo-url>

If you want to build and/or serve your site locally,
you also need to get local copies of the theme’s own submodules:

    git submodule update --init --recursive

#### API documentation

API documentation is generated using `Sphinx` with a theme
[`Read the Docs`](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html).
The `Read the Docs` theme is added as a submodule.
Install Sphinx ([learn more](https://www.sphinx-doc.org/en/master/index.html)).

    pip install -U Sphinx

If you want to add API documentation for third-party modules,
use the `intersphinx` extension, for proper operation,
install the required modules.
Learn more about [`intersphinx`](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html).

    cd <your local directory>/datumaro/
    pip install -r requirements.txt

Documentation is generated automatically from `rst` files and comments contained in the source code,
files located in `site/source/api` using
[`sphinx-autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
Comments in the source code should be in the format [reST](https://www.sphinx-doc.org/en/master/index.html),
[Google](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) or
[Numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#).

    sphinx-build -a -n site/source site/static/api

To describe new modules, you can generate the `rst` files using the
[`sphinx-apidoc`](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html).

    sphinx-apidoc -d 1 -o site/source/api datumaro

Sphinx generates documentation in html format in `site/static/api`.
After generating the documentation API,
you can generate a site with documentation.

#### Site generation

To build and preview your site locally, use:

    cd <your local directory>/datumaro/site/
    hugo server

By default, your site will be available at <http://localhost:1313/docs/>.

Instead of a "hugo server" command, you can use the "hugo" command
that generates the site into a "public" folder.

To build or update your site’s CSS resources you will need [PostCSS](https://postcss.org/)
to create final assets.
To install it you must have a recent version of [NodeJS](https://nodejs.org/en/)
installed on your machine,
so you can use npm, the Node package manager.
By default npm installs tools under the directory
where you run [npm install](https://docs.npmjs.com/cli/v6/commands/npm-install#description):

    cd <your local directory>/datumaro/site/
    npm ci

Then you can build a website in the "public" folder:

    hugo

[Read more](https://www.docsy.dev/docs/getting-started/)

### Update the submodule of the docsy theme

To update the submodule of the docsy theme you need to have a repository clone.
While in the repository folder,
use the git command:

    git submodule update --remote

Add and then commit the change to project:

    git add themes/
    git commit -m "Updating theme submodule"

Push the commit to project repo. For example, run:

    git push
