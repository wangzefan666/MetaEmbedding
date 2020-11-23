from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from parse import args
from utils import *
import tensorflow as tf
from model import Meta_Model


seed = 0
set_seed(seed)

# some pre-processing
num_words_dict = {
    'MovieID': 4000,
    'UserID': 6050,
    'Age': 7,
    'Gender': 2,
    'Occupation': 21,
    'Year': 83,
}
ID_col = 'MovieID'
item_col = ['Year']
context_col = ['Age', 'Gender', 'Occupation', 'UserID']

# training data of big ads: DataFrame
train = read_pkl("../data/big_train_main.pkl")
train_y = train['y']
train_x = train[[ID_col] + item_col + context_col]
train_t = pad_sequences(train.Title, maxlen=8)  # left padding
train_g = pad_sequences(train.Genres, maxlen=4)

# testing data of small ads
test_test = read_pkl("../data/test_test.pkl")
test_x_test = test_test[[ID_col] + item_col + context_col]
test_y_test = test_test['y'].values
test_t_test = pad_sequences(test_test.Title, maxlen=8)
test_g_test = pad_sequences(test_test.Genres, maxlen=4)

model = Meta_Model(ID_col, item_col, context_col, num_words_dict, model=args.model,
                   emb_size=args.emb_size, alpha=args.alpha,
                   warm_lr=args.lr, cold_lr=args.lr / 10., ME_lr=args.lr)

"""
Pre-train the base model, including look-up tables for ID, item and context and 
"""
batchsize = args.batch_size
n_samples = train_x.shape[0]
n_batch = n_samples // batchsize

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i_batch in range(n_batch):
    batch_x = train_x.iloc[i_batch * batchsize:(i_batch + 1) * batchsize]
    batch_t = train_t[i_batch * batchsize:(i_batch + 1) * batchsize]
    batch_g = train_g[i_batch * batchsize:(i_batch + 1) * batchsize]
    batch_y = train_y.iloc[i_batch * batchsize:(i_batch + 1) * batchsize].values
    loss, _ = model.train_warm(sess, batch_x, batch_t, batch_g, batch_y)

# evaluate origin cold start performance
test_pred_test = predict_on_batch(sess, model.predict_warm, test_x_test, test_t_test, test_g_test)
logloss_base_cold = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[pre-train]\n\ttest-test loss: {:.6f}".format(test_loss_test))
auc_base_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("\ttest-test auc: {:.6f}".format(test_auc_test))

# save pretrain model
save_path = saver.save(sess, args.saver_path)
print("Model saved in path: %s" % save_path)


'''
Train the Meta-Embedding generator
'''
minibatchsize = 20  # num of records of every ad: 20
batch_n_ID = args.batch_n_ID
batchsize = minibatchsize * batch_n_ID
for i_epoch in range(3):  # 可能轮流3次就够了
    # Read the few-shot training data of big ads
    # 对于每个ad，采样80个record，平均分到四个文件中，四个文件轮流作为小批量a和b
    if i_epoch == 0:
        _train_a = read_pkl("../data/train_oneshot_a.pkl")
        _train_b = read_pkl("../data/train_oneshot_b.pkl")
    elif i_epoch == 1:
        _train_a = read_pkl("../data/train_oneshot_c.pkl")
        _train_b = read_pkl("../data/train_oneshot_d.pkl")
    elif i_epoch == 2:
        _train_a = read_pkl("../data/train_oneshot_b.pkl")
        _train_b = read_pkl("../data/train_oneshot_c.pkl")
    elif i_epoch == 3:
        _train_a = read_pkl("../data/train_oneshot_d.pkl")
        _train_b = read_pkl("../data/train_oneshot_a.pkl")
    train_x_a = _train_a[[ID_col] + item_col + context_col]
    train_y_a = _train_a['y'].values
    train_t_a = pad_sequences(_train_a.Title, maxlen=8)
    train_g_a = pad_sequences(_train_a.Genres, maxlen=4)

    train_x_b = _train_b[[ID_col] + item_col + context_col]
    train_y_b = _train_b['y'].values
    train_t_b = pad_sequences(_train_b.Title, maxlen=8)
    train_g_b = pad_sequences(_train_b.Genres, maxlen=4)

    n_samples = train_x_a.shape[0]
    n_batch = n_samples // batchsize
    # Start training
    for i_batch in range(n_batch):
        batch_x_a = train_x_a.iloc[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_t_a = train_t_a[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_g_a = train_g_a[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_y_a = train_y_a[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_x_b = train_x_b.iloc[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_t_b = train_t_b[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_g_b = train_g_b[i_batch * batchsize:(i_batch + 1) * batchsize]
        batch_y_b = train_y_b[i_batch * batchsize:(i_batch + 1) * batchsize]
        loss_a, loss_b, _ = model.train_ME(sess,
                                           batch_x_a, batch_t_a, batch_g_a, batch_y_a,
                                           batch_x_b, batch_t_b, batch_g_b, batch_y_b, )
    # evaluate cold start performance
    test_pred_test = predict_on_batch(sess, model.predict_ME, test_x_test, test_t_test, test_g_test)
    logloss_ME_cold = test_loss_test = log_loss(test_y_test, test_pred_test)
    print("[Meta-Embedding]\n\ttest-test loss: {:.6f}".format(test_loss_test))
    auc_ME_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
    print("\ttest-test auc: {:.6f}".format(test_auc_test))

# save embedding generator
save_path = saver.save(sess, args.saver_path)
print("Model saved in path: %s" % save_path)

'''
Testing
'''
print('=' * 60)
print("COLD-START BASELINE:")
print("\t Loss: {:.4f}".format(logloss_base_cold))
print("\t AUC: {:.4f}".format(auc_base_cold))

test_a = read_pkl("../data/test_oneshot_a.pkl")
test_b = read_pkl("../data/test_oneshot_b.pkl")
test_c = read_pkl("../data/test_oneshot_c.pkl")

test_x_a = test_a[[ID_col] + item_col + context_col]
test_y_a = test_a['y'].values
test_t_a = pad_sequences(test_a.Title, maxlen=8)  # left padding
test_g_a = pad_sequences(test_a.Genres, maxlen=4)

test_x_b = test_b[[ID_col] + item_col + context_col]
test_y_b = test_b['y'].values
test_t_b = pad_sequences(test_b.Title, maxlen=8)
test_g_b = pad_sequences(test_b.Genres, maxlen=4)

test_x_c = test_c[[ID_col] + item_col + context_col]
test_y_c = test_c['y'].values
test_t_c = pad_sequences(test_c.Title, maxlen=8)
test_g_c = pad_sequences(test_c.Genres, maxlen=4)

minibatchsize = 20
batch_n_ID = args.batch_n_ID
batchsize = minibatchsize * batch_n_ID
test_n_ID = len(test_x_c[ID_col].drop_duplicates())
n_batch = test_n_ID // batch_n_ID

# train base-embedding for 3 epochs
saver.restore(sess, save_path)  # reload trained model
logloss_base_batch = []
auc_base_batch = []
for b in range(3):
    for i in range(n_batch):
        batch_x = test_x_a[i * batchsize:(i + 1) * batchsize]
        batch_t = test_t_a[i * batchsize:(i + 1) * batchsize]
        batch_g = test_g_a[i * batchsize:(i + 1) * batchsize]
        batch_y = test_y_a[i * batchsize:(i + 1) * batchsize]
        model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, embedding_only=True)

    # evaluate warm up performance
    test_pred_test = predict_on_batch(sess, model.predict_warm, test_x_test, test_t_test, test_g_test)
    test_loss_test = log_loss(test_y_test, test_pred_test)
    logloss_base_batch.append(test_loss_test)
    print("[baseline]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
        test_loss_test, 1 - test_loss_test / logloss_base_cold))
    test_auc_test = roc_auc_score(test_y_test, test_pred_test)
    auc_base_batch.append(test_auc_test)
    print("\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
        test_auc_test, test_auc_test / auc_base_cold - 1))

print("=" * 60)

# train meta-embedding for 3 epochs
saver.restore(sess, save_path)
logloss_ME_batch = []
auc_ME_batch = []
for b in range(3):
    for i in range(n_batch):
        batch_x = test_x_a[i * batchsize:(i + 1) * batchsize]
        batch_t = test_t_a[i * batchsize:(i + 1) * batchsize]
        batch_g = test_g_a[i * batchsize:(i + 1) * batchsize]
        batch_y = test_y_a[i * batchsize:(i + 1) * batchsize]
        # replace the original ID look-up table with embeddings produced by embedding generator
        if b == 0:
            aid = np.unique(batch_x[ID_col].values)
            for k in range(batch_n_ID):
                if k * minibatchsize >= len(batch_x):
                    break
                ID = batch_x[ID_col].values[k * minibatchsize]  # every ID has mini-batch records
                embeddings = model.get_meta_embedding(
                    sess, batch_x[k * minibatchsize:(k + 1) * minibatchsize],
                    batch_t[k * minibatchsize:(k + 1) * minibatchsize],
                    batch_g[k * minibatchsize:(k + 1) * minibatchsize],
                )
                emb = embeddings.mean(0)
                model.assign_meta_embedding(sess, ID, emb)
        model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, embedding_only=True)
    test_pred_test = predict_on_batch(sess, model.predict_warm, test_x_test, test_t_test, test_g_test)
    test_loss_test = log_loss(test_y_test, test_pred_test)
    logloss_ME_batch.append(test_loss_test)
    print("[Meta-Embedding]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
        test_loss_test, 1 - test_loss_test / logloss_base_cold))
    test_auc_test = roc_auc_score(test_y_test, test_pred_test)
    auc_ME_batch.append(test_auc_test)
    print("\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
        test_auc_test, test_auc_test / auc_base_cold - 1))


# write the scores into file.
res = [logloss_base_cold, logloss_ME_cold,
       logloss_base_batch[0], logloss_ME_batch[0],
       logloss_base_batch[1], logloss_ME_batch[1],
       logloss_base_batch[2], logloss_ME_batch[2],
       auc_base_cold, auc_ME_cold,
       auc_base_batch[0], auc_ME_batch[0],
       auc_base_batch[1], auc_ME_batch[1],
       auc_base_batch[2], auc_ME_batch[2]]

with open(args.log, "a") as logfile:
    logfile.writelines(",".join([str(x) for x in res]) + "\n")
