# 1003_Final

The link for training data can be found at https://www.vision.caltech.edu/datasets/cub_200_2011/

For Pretrian: Simply run train.py

For Fine tune python res.py -sp ft_set/support_set.pkl -tp ft_set/test_set.pkl -m models/emb_model_res18_cos.pth -s cosine python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l1 -e renyi python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model.pth -s l2 -e shannon

Create your own Support and Query(test) set: python create_support_test.py -i ./CUB_200_2011/images

python res.py -sp ./support_set.pkl -tp ./test_set.pkl -m models/emb_model_res18_triplet.pth -s l2
