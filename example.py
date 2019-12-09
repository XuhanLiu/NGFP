from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from models.dataset import MolData
from models.model import QSAR
import pandas as pd
import numpy as np
from models import util


def main(reg=False, is_extra=True):
    pair = ['TARGET_CHEMBLID', 'CMPD_CHEMBLID', 'PCHEMBL_VALUE',
            'CANONICAL_SMILES', 'ACTIVITY_COMMENT', 'STANDARD_TYPE', 'RELATION']
    df = pd.read_csv('data/AR_ALL.csv')
    cmps = df.set_index(pair[1])[pair[3]].drop_duplicates()
    df = df[pair].set_index(pair[0:2])
    df['PCHEMBL_VALUE'] = df.groupby(pair[0:2]).mean()
    numery = df[pair[2:4]].dropna().drop_duplicates()

    comments = df[(df.ACTIVITY_COMMENT.str.contains('Not Active') == True)]
    inhibits = df[(df.STANDARD_TYPE == 'Inhibition') & df.RELATION.isin(['<', '<='])]
    relations = df[df.STANDARD_TYPE.isin(['EC50', 'IC50', 'Kd', 'Ki']) & df.RELATION.isin(['>', '>='])]
    binary = pd.concat([comments, inhibits, relations], axis=0)
    binary = binary[~binary.index.isin(numery.index)]
    binary['PCHEMBL_VALUE'] = 3.99
    binary = binary[pair[2:4]].dropna().drop_duplicates()

    df = numery.append(binary)
    df = df[pair[2]].unstack(pair[0])
    df = df.sample(len(df))

    if reg:
        test = binary[pair[2]].sample(len(binary)).unstack(pair[0])
    else:
        df = (df > 6.5).astype(float)
        test = df.sample(len(df)//8)
        df = df.drop(test.index)
    data = df if is_extra else numery.sample(len(numery))

    indep_set = MolData(cmps.loc[test.index], test.values)
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE)
    folds = KFold(5).split(data)
    cvs = np.zeros(data.shape)
    inds = np.zeros(test.shape)
    out = 'output/gcn%s' % ('_' + subset if subset else '')
    for i, (trained, valided) in enumerate(folds):
        trained, valided = data.iloc[trained], data.iloc[valided]
        train_set = MolData(cmps.loc[trained.index], trained.values)
        valid_set = MolData(cmps.loc[valided.index], valided.values)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        net = QSAR(hid_dim=128, n_class=data.shape[1]).to(util.dev)
        net = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path='%s_%d' % (out, i))
        print('Evaluation of Loss in validation Set: %f' % net.evaluate(valid_loader))
        print('Evaluation of Loss in independent Set: %f' % net.evaluate(indep_loader))
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)

    data_score, test_score = pd.DataFrame(), pd.DataFrame()
    data_score['LABEL'] = data.stack()
    test_score['LABEL'] = test.stack()
    data_score['SCORE'] = pd.DataFrame(cvs, index=data.index, columns=data.columns).stack()
    test_score['SCORE'] = pd.DataFrame(inds, index=test.index, columns=test.columns).stack()
    data_score.to_csv(out + '.cv.txt')
    test_score.to_csv(out + '.ind.txt')


if __name__ == '__main__':
    BATCH_SIZE = 128
    N_EPOCH = 1000
    main()
