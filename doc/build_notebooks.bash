cd notebooks
# Convert notebooks to rst so we can convert them into html with sphinx's 
# style
ipython nbconvert *.ipynb --to rst
# When we write a sphinx-style intra-project link (ie :mod:`my_module`)
# in Ipython Notebook in a markdown box, and then convert to an rst, it ends
# up with double backticks (ie :mod:``my_module``). The below command changes
# it back so the link will work when we build the documentation
find . -type f -name '*.rst' -exec sed -E -i '' 's/:(mod|func|data|const|class|meth|attr|exc|obj):``(.*)``/:\1:`\2`/g' {} \;
