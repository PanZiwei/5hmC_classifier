1. Open VScode to get access to the sumner 
2. (initial) `git clone https://github.com/PanZiwei/5hmC_classifier.git`


1. Create a new repository on GitHub(https://github.com/PanZiwei/5hmC_classifier). Do not adde README, license or gitignore for now. Note the “remote repository URL”, which you will need in step 7 instead of **URL**.
2. Open VScode and log into sumner, open terminal
3. Initialize a new Git repository: `git init`
4. Create a `.gitignore` file
4. Stage your files before first commit: `git add.`
5. Commit the files: `git commit -m “initiate”`
6. Add remote repository: `git remote add origin **URL**`
7. Verify the remote repository: `git remote -v`
8. Push the first changes: `git push -u origin master`

For later usage:
```shell
git add --all #add current dir for repo change
git commit -m "message"
git push
```