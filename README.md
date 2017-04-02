# kaggle_memo
 This is my memorandum for kaggle competition. Sorry, only in Japanese. I'm not a profrssional data scientist, so the contents may be inaccurate.

## 特徴エンジニアリングについて 
- まずは、素うどんのXGBoostにかけて、plot_importance, feature_importances_を確認する。しかる後に、各特徴量をF-SCOREの高い順にExploratory Data Analysis (EDA)を行い、データに対する感覚を掴む。特徴量の数が少ないのであれば、初めからEDA。
- 情報を含まないcolumnsを除く。[Kaggle Kernel: [R](https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3), [Python](https://www.kaggle.com/yuansun/santander-customer-satisfaction/lb-0-84-for-starters)]
    - 標準偏差が0の説明変数 (constant cols) を除く。 
    - 重複した説明変数 (duplicated cols) を1つだけ残して他を除く。 
    - 相関係数が1である説明変数の組 (perfectly correlated cols) を探し、1つだけ残して他を除く。 
- 各列について、値が0である説明変数の数を数えて、合計値を追加の説明変数として加える (count 0 per row)。逆に0でないものの合計をとることもある。割と定石らしい。 
- カテゴリカル変数を、ダミー変数に変換する。 
- カテゴリカル変数を、目的変数のそのカテゴリにおける平均値（期待値）に変換する。これはLikelihood EncodingあるいはBayesian Encoding（Multinomial Naive Bayesにかけるのと同じなので）と呼ばれる手法である。 
Target Based Featureであるため、out-of-fold prediction (Leave-One-Out等) にしないとLeakしてしまうことがあるので注意が必要である。ただし、これによるLeakが気になるケースでは用いないほうが良いだろう。 
    - Likelihood Encodingは、1~数回しか観測されないようなレアなカテゴリ値が多数あると成り立たない。out-of-fold predictionにしないともろにLeakするし、out-of-fold predictionにすると今度は観測されなかったことになってしまう。Pseudo-observationを加えるという手もあるが、元の分布がそもそも統計的に取り扱えるような状態ではないために筋が良いとは言えない気がする。基本的には曜日などのように、どのカテゴリ値も十分な観測数があって、out-of-foldしようがしまいが分布に影響がほぼないようなケースでの適用が望ましいと思う。 
- 上記のような理由でLikelihood Encodingが適用できない場合、count encodingが有効な場合がある。これはそのカテゴリ値をとる頻度に変換する手法である（value_countsする）。 
- 主成分分析 (PCA) の上位いくつかを追加の特徴量として加える。PCAの追加は冗長に思えるが、これが有効となりうる説明としては、Kaggleの数値系コンペで主力となるXGBoostなどの決定木の眷属が軸に斜めの表現を不得手（表現できないわけではないのだが、PCAで回転した方が効率的）としていることに起因しているようだ。 
- 特徴が全く同じで目的変数が異なるペアをノイズとして除いても良い場合がある。 
- 多クラス分類において、各クラスに対する確率をナイーブベイズライクに見積もったものを追加する (Facebook Vの2nd, 3rd place solution)。 
- 時系列データにおいて、曜日や時刻などの周期性のある特徴量を円周上に配置し、cos, sinに分解することで、それらの循環連続性を表現する [[Kaggle Kernel](https://www.kaggle.com/zeroblue/facebook-v-predicting-check-ins/mad-scripts-battle-z/code)]。 
- 時系列データにおいて、曜日や時刻などの周期性のある特徴量に対して、例えば、{0:日曜日, 1:月曜日, 2:火曜日, 3:水曜日, 4:木曜日, 5:金曜日, 6:土曜日}のようになっているときに、{-2:金曜日, -1:土曜日, 0:日曜日, 1:月曜日, 2:火曜日, 3:水曜日, 4:木曜日, 5:金曜日, 6:土曜日, 7:日曜日, 8:月曜日}といった具合に両端をのばす（その分だけ列が増えることになる）。これにより、kNNなどで、両端の近さを表現しているようだ。 
- 時系列データにおいて、近いイベントほど重みを大きくするのが有効な場合がある。 
- 予測に必要な値（事前分布など）にXGBoostなどの予測値を代入する。 
- Expediaコンペのように各idと行き先の「座標」を勾配降下法などによって求め、得られる大圏距離を特徴量に追加するなどの手法がある。大掛かりな特徴エンジニアリングだが、Expediaでは極めて効果的だったようだ。 
- k-NearestNeighborsによって観測された近傍点の数を特徴量として追加する（Facebook Predict check-insの1st place Winner’s solution）。 
- 大規模だがローカリティの強いデータは分割して分散処理が有効である。その際はtraining dataのグリッドをオーバーラップさせると情報損失を抑えられる (Facebook V)。 
 
## Hyper Parameterの調整について 
- GridSearchCVを用いる。 
- RandomizedSearchCVを用いる。 
- hyperoptを用いる。 
 
## Ensembleについて 
- 何はともあれ、[KAGGLE ENSEMBLING GUIDE](http://mlwave.com/kaggle-ensembling-guide/)
- １層目の予測器群のCVによって生成したpredict_probaを特徴として用いる２層目の予測器群を構築する。元の特徴を併せて用いる場合と用いない場合の両方がある。これをStacked Generalizationと呼ぶ。初出は1992年の同名の論文であるが、Netflix PrizeやKDD Cup, Kaggle等の競技主導で発展した手法であり、現在の主なEnsemble方法らしい [[元論文](http://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf)], [[Code](https://github.com/nejumi/tools_for_kaggle/blob/master/stacked_generalization2.py)]。 
- Stackingと似たものにBlendingというものもある。シンプルで扱いやすいが、クロスしないのでデータの一部を捨てることになる。BlendingとStackingを同じ意味で用語として区別しないこともある。 
- 単体でPoorでもEnsembleで化けることがあるらしいので、捨てないこと。 
- Baggingによる改良が望めないか調べる。 
- 複数の予測器による予測の加重平均を取ることで、改良が望めないか調べる。Averagingは単純だが強力な手法である。XGBoostとRandomForestは相性が良いことが経験的に知られている。また、これらにNeural NetworkやSVM, kNNなどを加えるのが良い。これは、XGBoost/RandomForestはTree Baseのために決定境界が特徴軸に平行な矩形になるが、Neural Networkなどは曲線（曲面）となるため（よりdiverseな予測器群のensembleになるため）である。 
- 評価関数がMAP@kの場合、候補に重みを設定した上でSubmission同士のensembleが有効である。その場合、終盤のensembleに備えてMAP@j, j>kで予測・保存しておくと良い（ただし、可能であるならProbabilityのaveragingの方が当然好ましい） [[Kaggle Kernel](https://www.kaggle.com/c/facebook-v-predicting-check-ins/discussion/21265)]。 
- その他、論理和をとることが有効な場合がある。 
 
## Deep Learningについて 
- Deep Learning主体の画像系コンペ等は通常の数値データ系コンペとは多少やることが変わってくるため、コンペで得た知見とテクニックをここに追記していく予定。 
- 基本は、矩形回帰⇒キーポイント検出⇒アフィン変換による整列処理⇒分類で、いずれも教師あり学習。これはRight Whale RecognitionのWinning solutionでも同様であった。 
- 矩形回帰もmulti-box対応が必要になると急に難度が跳ね上がる。その場合はYOLO等の手法を参照のこと。 
- 矩形回帰やキーポイント検出に用いる座標データは、特に提供されない限りは自分で画像から作成する必要がある。これをannotationというが、そのための便利なannotation toolが色々公開されているので、利用すると良い (SlothやObject Marker等)。 
- ImageNetで予め学習済みのpre-trainedモデルを用いると良い。これを転移学習、(あるいはNNの場合は特にfine tuning)という。その際は、出力側の全結合層を当該課題に適したものに交換し、畳み込み層をフリーズしてAdamあたりで訓練⇒フリーズを出力側からちょっとずつ解除し、低learning rateのSGDあたりでじわじわ再訓練（fine tuning）する。いきなり全体を訓練すると畳み込み層の特徴抽出機能が崩壊するので注意。 
- 全結合層→畳み込み層の変換が有効だったりするらしい。 
- pre-trainedモデルは様々なものが公開されているので、それらのensembleを行うのが定石。代表的なところでは、Kerasに標準で実装されているものだけでも、VGG16, VGG19, InceptionV3, Xception, ResNet50等。 
- pre-trainedモデルの選択には、[arXiv:1605.07678](https://arxiv.org/abs/1605.07678)を参照すると良いかもしれない。InceptionV3あたりがバランス良い？ 
- Deep LearningはBaggingやAveragingの効果が特に大きいので、必ず行うこと。 
- Averagingを行う際に、全体を単純に加重平均するのではなく、columnごとに独立に重みを変えることでout-perform出来たとの報告あり(State Farm)。ただし、validation dataが小さいと過学習しやすいので注意である。
- Averagingは単純平均の他にも、N乗平均、対数平均など試してみる。
- 元画像の鏡像や回転・変形画像、ノイズやぼかしの付加によってデータを水増しする。これによって元画像のみで訓練するよりもロバストなモデルを構築できる。これをData Augmentationという。ただし、水増しデータがロバストの域を超えて外れ値になってしまうと、逆に精度を落としうるので注意。なお、KerasにはData Augmentationを逐次行いながら画像を生成してくれるImageDataGeneratorという便利で省メモリな機能がある。 
- 回帰による検出と分類を多段階にせず、一段階で高速に行う手法がある (Fast R-CNN, Faster R-CNN, YOLO, SSD等)。複雑なアーキテクチャであるが、おいおい理解すべきだろう。 
- 入力画像のサイズは、色々変えてみて人間の目で識別可能かどうかを試して見ると良い。ただし、人間の識別方法が最適でない場合もあるようで、上記の方法も確実ではないようだ。 
- 数値データと同様に画像系でもsemi-supervised learningは多くの場合で有効である。 
- テストデータに対するpseudo-labelingを行う場合にはk foldに分けて、予測対象以外のfoldに対するpseudo labelを用いる手法が有効らしい (Cross pseudo-labeling)。これは分類器がpseudo -labelの丸暗記を行うのを防ぐためであり、iwiwiがState Farmで用いた手法である。 
- 動画のキャプチャを課題に用いている場合、kNNで静止画から動画に復元できてしまう(State Farm)。 
- 単に高確信度データに対してだけpseudo-labelingするだけでなく、低確信度データまで含めてpseudo-labelingしてしまい、「とあるモデルによるpseudo-labelを予測するという事前学習」とみなしてモデルを学習しておく手法がある。これにより、目的のタスクに近い事前学習によるpre-trained modelを得られたのと同様のご利益が得られる可能性がある。得られた重みを初期値として、その後にちゃんと教師ありでfine-tuningする。 
- denoising autoencoderによる事前学習もよく知られている。ImageNetによるpre-trained modelが主流になってあまり使用されなくなってはいるようだ。 
- 活性化関数はReLUが標準的。PReLUは良いが、データ数が少ない場合には過学習しがちである（データから負側勾配を学習するので）。RReLUが性能良いらしいが、KerasでもChainerでも未実装である。[[arXiv](https://arxiv.org/abs/1505.00853)] 
- Deep learningでは特に過学習しやすく、その対策が重要である。出力側の構造を無駄に複雑にしない、dropout、batch normalization、early_stopping、正則化などで対処すること。 
- 予測時にテストデータに対してRandom 10 croppingを行い、Averagingすると良い。その際にmulti-scaleにするとより効果的である。 
 
## Leakageについて 
- 悲しいことであるが、KaggleにおいてもLeakageは度々発生している。機械学習のコンペとしては無意味で不毛な行為だが、こうしたLeakageに対処し、取り込むことを迫られる場合もある。 
- LeaderBoardのスコアに著しい不連続性が発生している場合、Leakage起因である可能性がある。 
- row_Idをfeatureに加えてみて、それが予測上意味を持ってしまっている場合、それ自身もLeakageであると同時に、より重大なLeakageの予兆である可能性がある。 
- 色々なfeatureの組み合わせでsort_valuesをしてみると見つかる可能性がある（Telstra, [BOSCH](https://www.kaggle.com/mmueller/bosch-production-line-performance/road-2-0-4/code)など）。 
- 通常の機械学習アルゴリズムでは、データ全体中での各行の統計的な分布の中での位置はモデル化してくれるが、前後の行との関係性や何行目にあるかなど、csvファイル中での空間的な位置に関する情報はモデルに反映されない。このタイプのLeakageはTelstraやTalkingData、BOSCHなどで確認されている。 
- ただし、前後の行との関係性に基づくFeatureを作る場合は注意が必要である。前後の行を2回参照すると自分自身を参照することになるため、特にTarget-based featureを用いている場合にはLeakする場合がある。
- モチベーション的に厳しいものがあるが、Data Exploration の練習と思って取り組むしかない。 
 
## その他 
- Neural NetworkやSVMなど、StandardScalarなどによってデータの標準化がされていないと精度が出ない予測器も多い。逆にXGBoostやRandomForest等はその辺は気にしていないようだ (木なので)。 
かといって、何も考えずに平均を引いてはならない。疎行列がいきなり密行列になってメモリに乗らなくなったりしかねない。 
- Public Leader Boardに過学習しかねないので注意のこと。自分のLocal CVを信じること。 あるいは、LocalとPublicをデータサイズよって加重平均するのも手。
- 大きすぎるLocal scoreの向上は手元でのLeakageの可能性大のため、気をつけること。そこを深堀りしてしまうと、大幅に時間を無駄にしてしまう。 
- 序盤はensembleによる安易なスコアアップは控えること。時間と余裕があるうちに、他の多くの人はやらないような独創的な手法を模索すべき。人と同じアプローチは後から取り入れれば良く、また終盤戦を勝ち残るための優位性になりえない。 
- 裏技チックではあるが、testデータに対するsemi-supervised learningは現状で主要な差別化ポイントの一つである。[Code](https://github.com/nejumi/tools_for_kaggle/blob/master/semi_supervised_learner.py)
    - ただし、昨今のKaggleではこれができてしまうこと自体が実課題に対応できないモデルを生む原因の一つとして問題視されており、今後は変わっていくと予想される。 
- EXCEL等でデータをスプレッドシートとして眺めて分析するのも、意外と有効である。重要箇所を色づけして眺めたり、targetの積算値を眺めたりしてパターンが無いか調べたりする（BOSCHコンペでの[radder氏の指摘](https://www.kaggle.com/c/bosch-production-line-performance/discussion/24065#138106)。EXCELも馬鹿にはできない）。 
- ダウンロードしたtrain.csv, test.csvの特徴量はあくまで主催者が用意したひとつの断面でしかない。最適であるとは限らないし、実際大抵はそうではない。 
