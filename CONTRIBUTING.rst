Thank you for considering contributing to ``pymc-learn``! Please read these
guidelines before submitting anything to the project.

Some ways to contribute:

- Open an issue on the `Github Issue Tracker <https://github.com/pymc-learn/pymc-learn/issues>`__. (Please check that it has not already been reported or addressed in a PR.)
- Improve the docs!
- Add a new machine-learning model. Please follow the guidelines below.
- Add/change existing functionality in the base function classes for ML.
- Something I haven't thought of?

Pull/Merge Requests
---------------------
To create a Pull Request against this library, please fork the project and work from there.

Steps
................

1. Fork the project via the Fork button on Github


2. Clone the repo to your local disk, and add the base repository as a remote.

   .. code-block:: bash

     git clone https://github/<YOUR-GITHUB-USERNAME>/pymc-learn.git
     cd pymc-learn
     git remote add upstream https://github.com/pymc-learn/pymc-learn.git

3. Create a new branch for your PR.

   .. code-block:: bash

     git checkout -b my-new-feature-branch

Always use a ``feature`` branch. It's good practice to never routinely work on the ``master`` branch.

4. Install requirements (probably in a virtual environment)

   .. code-block:: bash

     conda create --name myenv python=3.6 pip
     conda activate myenv
     pip install -r requirements.txt
     pip install -r requirements_dev.txt

   NOTE: On Windows, in your Anaconda Prompt, run ``activate myenv``.

5. Develop your feature. Add changed files using ``git add`` and then ``git commit`` files:

   .. code-block:: bash

     git add <my_new_model.py>
     git commit

to record your changes locally. After committing, it is a good idea to sync with the base repository
in case there have been any changes:

   .. code-block:: bash

     git fetch upstream
     git rebase upstream/master

Then push the changes to your Github account with:

   .. code-block:: bash

     git push -u origin my-new-feature-branch

6. Submit a Pull Request! Go to the Github web page of your fork of the ``pymc-learn`` repo. Click the 'Create pull request' button
to send your changes to the project maintainers for review. This will send an email to the committers.

Pull Request Checklist
................................

- Ensure your code has followed the Style Guidelines below
- Make sure you have written tests where appropriate
- Make sure the tests pass

   .. code-block:: bash

       conda activate myenv
       python -m pytest

   NOTE: On Windows, in your Anaconda Prompt, run ``activate myenv``.

- Update the docs where appropriate. You can rebuild them with the commands below.

   .. code-block:: bash

       cd pymc-learn/docs
       sphinx-apidoc -f -o api/ ../pmlearn/
       make html

- Update the CHANGELOG


Style Guidelines
.....................

For the most part, this library follows PEP8 with a couple of exceptions.

Notes:

- Indent with 4 spaces
- Lines can be 80 characters long
- Docstrings should be written as numpy docstrings
- Your code should be Python 3 compatible
- When in doubt, follow the style of the existing code

Contact
.............

To report an issue with ``pymc-learn`` please use the `issue tracker <https://github.com/pymc-learn/pymc-learn/issues>`__.

Finally, if you need to get in touch for information about the project, `send us an e-mail <devs@pymc-learn.org>`__.

Transitioning from PyMC3 to PyMC4
-----------------------------------

.. raw:: html

    <embed>
        <blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">.<a href="https://twitter.com/pymc_learn?ref_src=twsrc%5Etfw">@pymc_learn</a> has been following closely the development of <a href="https://twitter.com/hashtag/PyMC4?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC4</a> with the aim of switching its backend from <a href="https://twitter.com/hashtag/PyMC3?src=hash&amp;ref_src=twsrc%5Etfw">#PyMC3</a> to PyMC4 as the latter grows to maturity. Core devs are invited. Here&#39;s the tentative roadmap for PyMC4: <a href="https://t.co/Kwjkykqzup">https://t.co/Kwjkykqzup</a> cc <a href="https://twitter.com/pymc_devs?ref_src=twsrc%5Etfw">@pymc_devs</a> <a href="https://t.co/Ze0tyPsIGH">https://t.co/Ze0tyPsIGH</a></p>&mdash; pymc-learn (@pymc_learn) <a href="https://twitter.com/pymc_learn/status/1059474316801249280?ref_src=twsrc%5Etfw">November 5, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    </embed>