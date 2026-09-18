---
authors: balazskvancz
---

# Submitting assignments (GitHub)

We use the GitHub platform for submitting assignments. Each lab is submitted through its own GitHub repository, which you receive via the link in the assignment description. You must complete and upload your solution to the lab exercises in this repository. Once the finished solution has been uploaded (pushed) to the repository, it is submitted in the form of a so-called _pull request_ (PR), which you assign to your lab instructor.

!!! important "IMPORTANT"
    Following the process described here is mandatory. Solutions submitted in any other form will not be evaluated.

The process is as follows:

1. You must do your work in the GitHub repository created via the invitation link found in Moodle.

1. For the solution, create a separate branch named _solution_; do not work on _master_ / _main_. You may make any number of commits on this branch. Be sure to push the solution.

1. To submit, you must open a pull request named _labX_, where _X_ is the assignment number (for the first assignment: _lab1_, for the second assignment: _lab2_, etc.). The pull request's source is the solution branch, and its target is the original main branch (_master_ / _main_). You must assign the pull request to your lab instructor. You can find your lab instructor's GitHub username in Moodle. Leave the pull request open.

1. Check whether the _automated check_ (see below) found any errors. If it did, fix them and push to the solution branch.

1. If you have a question about the result or the evaluation, you can ask in a pull request comment. To notify the lab instructor, use the `@username` mention in the comment text.

Help with using git and GitHub:

- [GitHub git documentation](https://docs.github.com/en/get-started) - especially the "Using Git" section

- [git learning resources](https://git-scm.com/learn)


## Changes to the assignment

It may happen that the issued assignment changes mid-course, after the individual repository has been created. The instructors send out the changes: a pull request will appear in your repository. The PR's target branch defaults to the main branch, but the changes must also appear in the solution. If you haven't created the solution branch yet, the PR can simply be merged in. If you have already created the solution branch, [change the PR's target branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-base-branch-of-a-pull-request) to the solution branch, then merge the PR. If the change ended up on the wrong branch and therefore didn't make it into the solution branch, you can still bring the changes over to the solution branch afterwards, either via a pull request or via a merge operation without a pull request.


## Automated checks

For evaluating the lab assignments, we also rely on [GitHub Actions](https://github.com/features/actions). It lets us run operations and programs on git repositories. In this course, we only perform simple checks with it, such as verifying the presence of neptun.txt.

You will receive a notification in the pull request about the completed evaluation. If you'd like to take a closer look at what happened behind the scenes, or for example view the application logs, you can [get started](https://docs.github.com/en/actions/how-tos/monitor-workflows/view-workflow-run-history) under _Actions_ on the GitHub interface.

More information about GitHub Actions can be found [in the official documentation](https://docs.github.com/en/actions).
