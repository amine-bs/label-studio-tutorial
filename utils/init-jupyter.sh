WORK_DIR=/home/onyxia/work
CLONE_DIR=${WORK_DIR}/repo-git

# Clone course repository
REPO_URL=https://github.com/amine-bs/label-studio-tutorial.git
git clone --depth 1 $REPO_URL $CLONE_DIR

# Put chapter data in the training dir
cp ${CLONE_DIR}/tutorial.ipynb ${WORK_DIR}/
cp -r ${CLONE_DIR}/notebook-img ${WORK_DIR}/

# Give write permissions
chown -R onyxia:users $WORK_DIR/

# Remove course Git repository
rm -r $CLONE_DIR

# Open the relevant notebook when starting Jupyter Lab
jupyter server --generate-config
echo "c.LabApp.default_url = '/lab/tree/tutorial.ipynb'" >> /home/onyxia/.jupyter/jupyter_server_config.py