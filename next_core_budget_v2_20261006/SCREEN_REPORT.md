# MiniFly budget v2 screen

固定字节接口与既有 Full151 机制；不扩展任务、样本或候选。

| 配置 | n | old E1 | new E1 | revision E1 | reuse W | reuse W−N | W online CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CENTER | 8 | 0.9271 | 0.9688 | 0.4375 | 0.6667 | 0.1250 | 81.5276 |
| ERROR | 8 | 0.9583 | 0.9961 | 0.9688 | 0.6875 | 0.1250 | 82.4965 |
| P005 | 8 | 0.9583 | 0.9961 | 0.9844 | 0.6250 | 0.0833 | 84.3161 |
| P0005 | 8 | 0.9583 | 0.9961 | 0.9844 | 0.6875 | 0.1250 | 85.8099 |
| S3_CUE | 8 | 0.9583 | 0.9961 | 0.9844 | 0.7083 | 0.1667 | 115.8637 |
| S3_RAND | 8 | 0.7604 | 0.8281 | 0.7031 | 0.7083 | 0.1458 | 118.2776 |
| REL05 | 8 | 0.9479 | 0.9922 | 0.9062 | 0.7083 | 0.1875 | 144.8873 |
| REL10 | 8 | 0.8802 | 0.9531 | 0.8750 | 0.7292 | 0.2083 | 144.5287 |
| REL20 | 8 | 0.8281 | 0.9375 | 0.8281 | 0.7083 | 0.1875 | 143.2637 |
| REL_PERM10 | 8 | 0.9427 | 0.9961 | 0.9688 | 0.6875 | 0.1250 | 145.5849 |
| FIRST10 | 8 | 0.9583 | 0.9766 | 0.8281 | 0.7083 | 0.1667 | 145.0217 |
| Q_HALF | 8 | 0.9583 | 0.9961 | 0.9688 | 0.7917 | 0.2292 | 82.4984 |

裁决：`SELECTED_ONE_CANDIDATE`。

E1 是教过键的回忆/保留；E3 是未强化关系的有限复用。精确键查表可解决 E1，故 E1 单独通过不证明 E3、推理或自主学习。所有测试仍采用提示输出与到来的答案字节教学。筛选结果是探索性的；确认只检验预先锁定的一组候选和目标。

完整收据/不可用标记：176；所有配置均保留。

```json
{
  "verdict": "SELECTED_ONE_CANDIDATE",
  "winner": {
    "configuration": "Q_HALF",
    "parent": "ERROR",
    "target": "reuse",
    "delta": 0.05,
    "gain": 0.10416666666666666,
    "eta_gain": 22.28732187658263,
    "R": 2.0832853652467334,
    "normalized_gain": 2.083333333333333,
    "total_online_W_CPU": 82.49842650000022
  },
  "ranked": [
    {
      "configuration": "Q_HALF",
      "parent": "ERROR",
      "target": "reuse",
      "delta": 0.05,
      "gain": 0.10416666666666666,
      "eta_gain": 22.28732187658263,
      "R": 2.0832853652467334,
      "normalized_gain": 2.083333333333333,
      "total_online_W_CPU": 82.49842650000022
    }
  ],
  "guards": {
    "ERROR": {
      "passes": true,
      "reasons": [],
      "CPU_ratio": 1.01188460045999,
      "harms": [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0
      ]
    },
    "Q_HALF": {
      "passes": true,
      "reasons": [],
      "CPU_ratio": 1.000023025211716,
      "harms": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    },
    "P005": {
      "passes": false,
      "reasons": [
        "reuse_protection_reuse_W"
      ],
      "CPU_ratio": 1.0220562391068861,
      "harms": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    },
    "P0005": {
      "passes": true,
      "reasons": [],
      "CPU_ratio": 1.0401641680624902,
      "harms": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    },
    "S3_CUE": {
      "passes": true,
      "reasons": [],
      "CPU_ratio": 1.4044677495938658,
      "harms": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ]
    },
    "REL05": {
      "passes": false,
      "reasons": [
        "parent_protection_revision",
        "online_CPU_ratio"
      ],
      "CPU_ratio": 1.7562842084855237,
      "harms": [
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0
      ]
    },
    "REL10": {
      "passes": false,
      "reasons": [
        "accuracy_old",
        "parent_protection_old",
        "CENTER_protection_old",
        "parent_protection_new",
        "accuracy_revision",
        "parent_protection_revision",
        "online_CPU_ratio"
      ],
      "CPU_ratio": 1.7519373497383717,
      "harms": [
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1
      ]
    },
    "REL20": {
      "passes": false,
      "reasons": [
        "accuracy_old",
        "parent_protection_old",
        "CENTER_protection_old",
        "parent_protection_new",
        "CENTER_protection_new",
        "accuracy_revision",
        "parent_protection_revision",
        "online_CPU_ratio"
      ],
      "CPU_ratio": 1.7366023041794107,
      "harms": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    }
  },
  "alphas": [
    0.03,
    0.01,
    0.01
  ],
  "screen_worlds": [
    61008001,
    61008002,
    61008003,
    61008004,
    61008005,
    61008006,
    61008007,
    61008008
  ],
  "confirmation_roster": [
    "CENTER",
    "ERROR"
  ],
  "execution_source_identity": "3617e574074d3555ab3b0d2d6571effbd2dcb6d6269f68cc84baa3d8759f34b2",
  "plan_sha256": "bcc249c79ee623f80ae9487f630cd8e408381f858bdc93724f9131a3b0f86b3f",
  "screen_receipts": [
    {
      "file": "61008001_CENTER_lifetime.json.gz",
      "sha256": "59b9f5a5e0f9675ed22d6bd5d0a0fcca387f9ae0e734bb535ba8cd24ba209ff4"
    },
    {
      "file": "61008001_CENTER_reuse.json.gz",
      "sha256": "ff854d6ea6e6087aff4a25433c9109e1776dfd211cf0b76caea7d008fabc07a3"
    },
    {
      "file": "61008001_ERROR_lifetime.json.gz",
      "sha256": "8b93b08b24cbd6b4b60ddd944f3cb2c9892aa7ac20ffc7952447d06397fe5a7f"
    },
    {
      "file": "61008001_ERROR_reuse.json.gz",
      "sha256": "e09ee4ec3751678b3baab30e95b9e30b1a6ac54ff953d9f0434e2e287a1ebdd5"
    },
    {
      "file": "61008001_P005_lifetime.json.gz",
      "sha256": "1313b1d09c00a3802cf3338b497b6bb009b8f1ed37fb693bf0cb12d6f0cc7428"
    },
    {
      "file": "61008001_P005_reuse.json.gz",
      "sha256": "22c0899874ba7214e02e8b0eec7cee82bbd613d60280bab6934afd18c14b31e5"
    },
    {
      "file": "61008001_P0005_lifetime.json.gz",
      "sha256": "6740566bcb292c2dff1e4e96743f120244252509721f7152a932304a8a0c890c"
    },
    {
      "file": "61008001_P0005_reuse.json.gz",
      "sha256": "bd2390614b5d2b7762a5df9c3385b20e69ac4bb41b80aa94e3e1f7d3ea623786"
    },
    {
      "file": "61008001_S3_CUE_lifetime.json.gz",
      "sha256": "16407146d0371cd5cdee5adddab7f1489b3a27e3aca5bec3bd12d279c392a848"
    },
    {
      "file": "61008001_S3_CUE_reuse.json.gz",
      "sha256": "9e024260376b1a8fb5eaa337552b4c7efb38b7a2d61d6776cd67cb3550dc5cb5"
    },
    {
      "file": "61008001_S3_RAND_lifetime.json.gz",
      "sha256": "c391797cb44e77e6214494a40ac15836d0062ab9f56eabba2d6d574b69938359"
    },
    {
      "file": "61008001_S3_RAND_reuse.json.gz",
      "sha256": "a673ed2ccc85cda63c2a71475500ea82cd5c0b6ccf1556ead05bab3a45adf11a"
    },
    {
      "file": "61008001_REL05_lifetime.json.gz",
      "sha256": "5099487353c5ae603e49693733675eda9de0819963bfa08490a251089043d5a1"
    },
    {
      "file": "61008001_REL05_reuse.json.gz",
      "sha256": "416a7613907afa26572309b73bc6337680c8c09470e9debc30bca9ed82f23560"
    },
    {
      "file": "61008001_REL10_lifetime.json.gz",
      "sha256": "4d08119aa97a71d76b8515559d507d0af37cb52c2f71c99d4b5a86fca73414bb"
    },
    {
      "file": "61008001_REL10_reuse.json.gz",
      "sha256": "a1e15362f90a87a2410102fc26ee365a3b15f65416421217632a02a27ce7fb74"
    },
    {
      "file": "61008001_REL20_lifetime.json.gz",
      "sha256": "db827c68ca822f66d53a866cd1d89c6438a920bb30e129d5b5652dccc1c01bb9"
    },
    {
      "file": "61008001_REL20_reuse.json.gz",
      "sha256": "31495d48ee920e5acb4c1de422569d34421915f101d11a105163d96252e85098"
    },
    {
      "file": "61008001_REL_PERM10_lifetime.json.gz",
      "sha256": "21804c4d2ad74e65d12af5114fcb1c8c854834c8bf3c34311581b4adf751ad61"
    },
    {
      "file": "61008001_REL_PERM10_reuse.json.gz",
      "sha256": "c5d27609b041533d0998daa2f9a2be546a238866fb5097464487050de58d7cf7"
    },
    {
      "file": "61008001_FIRST10_lifetime.json.gz",
      "sha256": "31a237cb8c10cf27c22aa392144338b4fe5e896d5a925104868253b72e65084c"
    },
    {
      "file": "61008001_FIRST10_reuse.json.gz",
      "sha256": "1260abe25fa6cf4be03e943b6945ccd523c9b5e24586f19f37cda399ebe1e7bf"
    },
    {
      "file": "61008002_CENTER_lifetime.json.gz",
      "sha256": "e4939c286c571a8a47817ccd82688cc2793db8a71b3dbd219ef8f97a5c68a123"
    },
    {
      "file": "61008002_CENTER_reuse.json.gz",
      "sha256": "86aeb589bfcc4990d4a19393659b77a77703d2f427e6c19608f0561feb139e18"
    },
    {
      "file": "61008002_ERROR_lifetime.json.gz",
      "sha256": "8de588df6969369af4ef5f8f08fa2c6202db388e92dfdfee8a6c593c0c6cc67f"
    },
    {
      "file": "61008002_ERROR_reuse.json.gz",
      "sha256": "3d2e676a506a1c20dded000a004ba8bc5433e1d2e4b721560ffcb4930adf9687"
    },
    {
      "file": "61008002_P005_lifetime.json.gz",
      "sha256": "aa5888b1ecec76279c8879dec0ff381909308e6315b06053a8bda6a8969c38bf"
    },
    {
      "file": "61008002_P005_reuse.json.gz",
      "sha256": "0c3ded99338400c08dac93a55b3f1bae380a7c6b161b65c3063cfc0ccb4038b7"
    },
    {
      "file": "61008002_P0005_lifetime.json.gz",
      "sha256": "98884d820d199e1a80fe62ac9b8ebe7a0eeb2562d0c17ac25371dd3fd1ba8d7b"
    },
    {
      "file": "61008002_P0005_reuse.json.gz",
      "sha256": "daea2e149377b9def012116f66671ba15eba90916df5bdba6624327ca0ea5387"
    },
    {
      "file": "61008002_S3_CUE_lifetime.json.gz",
      "sha256": "06e601179ffe136c75e0773ed3bce63c6f6bb8c13438911b833cb46ee9fdbf94"
    },
    {
      "file": "61008002_S3_CUE_reuse.json.gz",
      "sha256": "ad8e6feeb93edf022c9d372338c6e37c1c86a434c4a387f0be01b8f53a4fd1cf"
    },
    {
      "file": "61008002_S3_RAND_lifetime.json.gz",
      "sha256": "0ebbb1204b2a8ab93564e034e3d8ab48782285aa41bac4677915c150692a4e01"
    },
    {
      "file": "61008002_S3_RAND_reuse.json.gz",
      "sha256": "d5b932ea8b59255f684ad59250430c2d5c11239d99acd1e0482d8e53c6628dc7"
    },
    {
      "file": "61008002_REL05_lifetime.json.gz",
      "sha256": "1284b4b6d86226d960fae3c82af1a491fb607028fa4524abdd89197af59eba6b"
    },
    {
      "file": "61008002_REL05_reuse.json.gz",
      "sha256": "8925cd60f8b322c43b9963431825f51319d3d5b9993454d93829233693bcdeb1"
    },
    {
      "file": "61008002_REL10_lifetime.json.gz",
      "sha256": "1e13c02eb5ff96aeffa0c4acd674f09cd626e3a1840cb1e8b2acff449e60f1e8"
    },
    {
      "file": "61008002_REL10_reuse.json.gz",
      "sha256": "89302d050d09e5bc1f0812c1d40ecfdcb2e3222e63690c511cdbb77ca6faa078"
    },
    {
      "file": "61008002_REL20_lifetime.json.gz",
      "sha256": "dca380f0c577e8dbbc5e1ab724f6f4149823d3bdc454793bf180bacf70dd18aa"
    },
    {
      "file": "61008002_REL20_reuse.json.gz",
      "sha256": "ea0cf3a27586e8ce63f5959a781b1b0046d81245d431d2522c8b624e90aa9dee"
    },
    {
      "file": "61008002_REL_PERM10_lifetime.json.gz",
      "sha256": "399f5c9f7f8474797866bc527e864a875d2fbc14c1cfee6ca800f9b30bbe8f0a"
    },
    {
      "file": "61008002_REL_PERM10_reuse.json.gz",
      "sha256": "07fecd2111707c356663872e3bd68f9c7d8e00aef6e517539b5cad2e9152d62e"
    },
    {
      "file": "61008002_FIRST10_lifetime.json.gz",
      "sha256": "8a66195098ff75985ccdf953de8f692136399b21daa6c3cf4c66c9e49afab32a"
    },
    {
      "file": "61008002_FIRST10_reuse.json.gz",
      "sha256": "78a0e8fbf35a56d4ec813b01e4212dc323e885b5f003c7519bd7da5a8cb05f6e"
    },
    {
      "file": "61008003_CENTER_lifetime.json.gz",
      "sha256": "4d097c812e61f2da60157c44f1fb46fb9ab4ad854b8cc801a5c1bc5c7c4cabad"
    },
    {
      "file": "61008003_CENTER_reuse.json.gz",
      "sha256": "0c001ec515abad5c6712f466947d070fbf184bc5524c6e71f5f2ab9d9a9f2c48"
    },
    {
      "file": "61008003_ERROR_lifetime.json.gz",
      "sha256": "1ae14f2579d74db4cb52935589e31edbc9ccf4410f72d80e3160b1116e161bcc"
    },
    {
      "file": "61008003_ERROR_reuse.json.gz",
      "sha256": "a39a12850de2a0e1c18e3b2f15ba2c24b675a8f746e3647c2044a32f6730df62"
    },
    {
      "file": "61008003_P005_lifetime.json.gz",
      "sha256": "29e2c5a8d6b8ead91e94f20c96cf84f39b7b17b9d207e8c048575b83363890be"
    },
    {
      "file": "61008003_P005_reuse.json.gz",
      "sha256": "b81452958450d6d0680f7662fee0764c3bfa564b314f7284a9ed99c8a7106e76"
    },
    {
      "file": "61008003_P0005_lifetime.json.gz",
      "sha256": "06dda6d883d4c28c1c539875d5564d8996ec767997015d32d8189006999253ea"
    },
    {
      "file": "61008003_P0005_reuse.json.gz",
      "sha256": "8a65bb306a1471f7595d2c114eb539203bcd58a6e68d9f73173c02b0ecb103df"
    },
    {
      "file": "61008003_S3_CUE_lifetime.json.gz",
      "sha256": "c8d53a270ec938289b5d0af4834efc2af11d8dd517ab273ac1feb26102586e6c"
    },
    {
      "file": "61008003_S3_CUE_reuse.json.gz",
      "sha256": "81267fc416eda07be33128b026b6604e35c4dcfa0faffb3ca45e5690011d1784"
    },
    {
      "file": "61008003_S3_RAND_lifetime.json.gz",
      "sha256": "fb55cbcefeaa66b03295415af77cf30c37291a3c33b150902da267846c0c7a04"
    },
    {
      "file": "61008003_S3_RAND_reuse.json.gz",
      "sha256": "cc384b4801597561aee8f63b380cdfc27277001acc34f44f4d5db74c83c77af5"
    },
    {
      "file": "61008003_REL05_lifetime.json.gz",
      "sha256": "48d010a5bbe5c13cdcc027f869ee1cc3a91cd656d47d7d4d75e5ec6aadadd294"
    },
    {
      "file": "61008003_REL05_reuse.json.gz",
      "sha256": "cdb23546c0717daeb59f80a6d23f6a09cb3c81484908d4c7379cdbe3b2b278a6"
    },
    {
      "file": "61008003_REL10_lifetime.json.gz",
      "sha256": "436d29f07000859ceeb7eb7471439eb80882cd4d653ac08d1b3b68b99aa0a695"
    },
    {
      "file": "61008003_REL10_reuse.json.gz",
      "sha256": "8f41bd8ab87c57a59cbf95cac0ba09ea2da29c46cb8f91db415e1051aadcdf44"
    },
    {
      "file": "61008003_REL20_lifetime.json.gz",
      "sha256": "37c55fdc13820c4f3f684088a32fd33109254fcb6b28ea25338ee0fc9343a28f"
    },
    {
      "file": "61008003_REL20_reuse.json.gz",
      "sha256": "4d9092676b936c4cc2fc66dbdc4e423e2b1f56897aebfcdb9a288909d4a7c414"
    },
    {
      "file": "61008003_REL_PERM10_lifetime.json.gz",
      "sha256": "49f15d56faf24eb3f08718ae2d26a2211520a4817539e49ba26c03cb652c5b1d"
    },
    {
      "file": "61008003_REL_PERM10_reuse.json.gz",
      "sha256": "663f23c09fa1bb19b45ba9ae6dfa474c5fdb272eee22131070ec0baf169b8afb"
    },
    {
      "file": "61008003_FIRST10_lifetime.json.gz",
      "sha256": "a9093ba8485bf5e19daa007dcf2e85df30700fbadc6394282cabc57d5edc58c3"
    },
    {
      "file": "61008003_FIRST10_reuse.json.gz",
      "sha256": "16796efefca1df3a54305d9e503c02b8c4029a707c53520a3fba81b9a0aac186"
    },
    {
      "file": "61008004_CENTER_lifetime.json.gz",
      "sha256": "818a64a97d7e451a4051c1689e23541c18097dc83a093e94380792c3665a1230"
    },
    {
      "file": "61008004_CENTER_reuse.json.gz",
      "sha256": "690ed60e8f5291732f90f79c4f61718db098abf3d2d756d57e798f4438901682"
    },
    {
      "file": "61008004_ERROR_lifetime.json.gz",
      "sha256": "05b7274c626a5e92b67f1e683031ffd5e2443c14011af12f69bb6bb229a867f7"
    },
    {
      "file": "61008004_ERROR_reuse.json.gz",
      "sha256": "6f8b9d896b3b4c68baa714d9b4f4d9b617bc3d6e277a955f85ea62bbe8d8e5b9"
    },
    {
      "file": "61008004_P005_lifetime.json.gz",
      "sha256": "c666208b4cbe9771677ff61c24e76246137d5873497373cf9c3e78cef0293105"
    },
    {
      "file": "61008004_P005_reuse.json.gz",
      "sha256": "84fb5823acc18a5a0c69e48d8fcb074ff2d8803505585b323ebbe3f953b47348"
    },
    {
      "file": "61008004_P0005_lifetime.json.gz",
      "sha256": "ac0e016743d738645ae2586e0a6cc51c8ddbc8327fc81427ed65cf1fb5af10d9"
    },
    {
      "file": "61008004_P0005_reuse.json.gz",
      "sha256": "df6dac667fe3f6b5063590de27fc0792daa2ac4f83e1c83f32e8f0a91b9c3a36"
    },
    {
      "file": "61008004_S3_CUE_lifetime.json.gz",
      "sha256": "1ef08f6b11e9a77e66dd13f4dbdb58c04c679482901c257815f627d1c593a53c"
    },
    {
      "file": "61008004_S3_CUE_reuse.json.gz",
      "sha256": "c710d2db01cfd037ae56a23c58f9f850b92e14e66b4eaf24bc2bdc62c977c1e5"
    },
    {
      "file": "61008004_S3_RAND_lifetime.json.gz",
      "sha256": "751c73e52dd2fdd48f01abefbbefaca0f1ea92f57e0b2c5643f77a1562b8cca0"
    },
    {
      "file": "61008004_S3_RAND_reuse.json.gz",
      "sha256": "aa7028b11d3cfdabf4676fdc28dc2279fc1b178ca6b662dda0511ff36cde7403"
    },
    {
      "file": "61008004_REL05_lifetime.json.gz",
      "sha256": "db7fa1f000a2831c7e3bbe302c56291b97cb2503cf1c3bb32f0d76ae5bd95c80"
    },
    {
      "file": "61008004_REL05_reuse.json.gz",
      "sha256": "8dd0f068b50ac512bab4ca2fac7e8f4eee1382bad9d27539ea1ebb8bcff36a34"
    },
    {
      "file": "61008004_REL10_lifetime.json.gz",
      "sha256": "9323e8749a61fc3218db3c62b1a43bdc2fa9c43b9479650e8b64f0c2f2a7c06b"
    },
    {
      "file": "61008004_REL10_reuse.json.gz",
      "sha256": "f01d24a20fdefef00432c87389b7bae9539958982f7a4099c055aa088d51f44b"
    },
    {
      "file": "61008004_REL20_lifetime.json.gz",
      "sha256": "ee3bb030fa9b3016311f7a6e15e7ee25018f8b221b86d9f1f25a0288c6600f40"
    },
    {
      "file": "61008004_REL20_reuse.json.gz",
      "sha256": "011eac7a88b268c4a370fbce03b880256c11039b71123f997bf9f93de5920fb4"
    },
    {
      "file": "61008004_REL_PERM10_lifetime.json.gz",
      "sha256": "9bf2209c965f3942119f2e1fc8e20bf62600822c65bdafef23ee318ec7c3afe8"
    },
    {
      "file": "61008004_REL_PERM10_reuse.json.gz",
      "sha256": "85ede2bc5271c295e1a9223ed226f28182828ce68d35f84001e66e6d5467e5d5"
    },
    {
      "file": "61008004_FIRST10_lifetime.json.gz",
      "sha256": "b0a5efdd65c6d51efd39fdcfb1d15d29a5fb2fc6a79966b3d6efdef028d6b494"
    },
    {
      "file": "61008004_FIRST10_reuse.json.gz",
      "sha256": "85e640ff2c8057ebfd6762d9cc9efc01658ebac7f59d85a03166cc9f928bd0a8"
    },
    {
      "file": "61008005_CENTER_lifetime.json.gz",
      "sha256": "67a1389c0d1146b4ff1a26b2465f8376d97adb3304457bb764e1d9b77b3c4098"
    },
    {
      "file": "61008005_CENTER_reuse.json.gz",
      "sha256": "4da8d2418d207fd62f932c65d425093070df26b2781e7b517d2d8657646ade75"
    },
    {
      "file": "61008005_ERROR_lifetime.json.gz",
      "sha256": "6aa0caf73efe2f0f6001357779fe3a1538e2c30724f653fbbc18b52610a5ab15"
    },
    {
      "file": "61008005_ERROR_reuse.json.gz",
      "sha256": "38a3e2d0ec53c42cd2d22cf04a028496ac7a659ab6cf0e6eeee5c5b8bba079f3"
    },
    {
      "file": "61008005_P005_lifetime.json.gz",
      "sha256": "cd5eb02fd74d4217b95a04144d8bd89d3d81fc56d6e05eaa09aa60dc9fec63a7"
    },
    {
      "file": "61008005_P005_reuse.json.gz",
      "sha256": "d0c85fa4d41691d3ad9ec17260859f4d71ff3c6c1ff8a4b551f96929af43e80a"
    },
    {
      "file": "61008005_P0005_lifetime.json.gz",
      "sha256": "070e6f8030af33f520aa63e82cac4dd338813edb7d969607e98ef82388bd52c4"
    },
    {
      "file": "61008005_P0005_reuse.json.gz",
      "sha256": "f5b4423e83ad73771776c2b6b1a18c8eb2d60e7bcfa2a3c0de679e9ce88e4c9d"
    },
    {
      "file": "61008005_S3_CUE_lifetime.json.gz",
      "sha256": "c908c71ad9c775c451633885149bd7f18d60a22e05e5540265553a056306eccc"
    },
    {
      "file": "61008005_S3_CUE_reuse.json.gz",
      "sha256": "3d4464fa3fcdf4e700fc2bc8edb0630a0e632bd234165ba16cee57b7f819baba"
    },
    {
      "file": "61008005_S3_RAND_lifetime.json.gz",
      "sha256": "95f6e373cd3037f0600a12d181e6422902ca23941d7e7fab748b408b0c545a77"
    },
    {
      "file": "61008005_S3_RAND_reuse.json.gz",
      "sha256": "6cdc57fa3695de04bfdcb4703fd20e6e5ee2accb0d25986497a7061929bafb84"
    },
    {
      "file": "61008005_REL05_lifetime.json.gz",
      "sha256": "db645e63c523034ca2c143f59b0edfbc0a348cb664f4fd54e9bc4cb010ab5f75"
    },
    {
      "file": "61008005_REL05_reuse.json.gz",
      "sha256": "23774f797c2945435760fb77cf9436423b760f1d2db6b0eafd2bb3d5bbd0e6fe"
    },
    {
      "file": "61008005_REL10_lifetime.json.gz",
      "sha256": "4084203ef0f7aa19d533876ee06c37d0d734eb45b9f7d4ac1cf070b9ff422c59"
    },
    {
      "file": "61008005_REL10_reuse.json.gz",
      "sha256": "39e3d48d821df2088d75ba8145c78c0b3432edbc059cc6cbea1952bf91a2c12f"
    },
    {
      "file": "61008005_REL20_lifetime.json.gz",
      "sha256": "06453279c615f4b7e6293496dd4d5cf97838772f49cf3e4fc27dd474dd12b7ff"
    },
    {
      "file": "61008005_REL20_reuse.json.gz",
      "sha256": "b7b97f5b9519b130a2ebb3067118db1e6a5d9ec8c9c65032f575c981b1aaece2"
    },
    {
      "file": "61008005_REL_PERM10_lifetime.json.gz",
      "sha256": "7afd8900f662c0a211f58a3b102379eadc36c7abedf603d0ea6d0ace0a97defd"
    },
    {
      "file": "61008005_REL_PERM10_reuse.json.gz",
      "sha256": "0729870ec45dc634e01ecb64ae74880f03fb9e425c3f6bda8f18276550f8c897"
    },
    {
      "file": "61008005_FIRST10_lifetime.json.gz",
      "sha256": "10b3971523d38a90b894ad599c31a5acfa0a7cf01ef4d4aedc78861db79cec87"
    },
    {
      "file": "61008005_FIRST10_reuse.json.gz",
      "sha256": "de31f793fa871a5ae93651a05ddc9e01316aac3f45fcf0ec438492bf1c58579d"
    },
    {
      "file": "61008006_CENTER_lifetime.json.gz",
      "sha256": "4d244127a8bbfb791847d54cf53a8da99d3c34774d1732ea52888b4c6c445b5e"
    },
    {
      "file": "61008006_CENTER_reuse.json.gz",
      "sha256": "6c621689d76ab958ee9a0639ca148e7cdfab922559cfdda2be32b1d750a5ffad"
    },
    {
      "file": "61008006_ERROR_lifetime.json.gz",
      "sha256": "acd37b22b88e0694409af24122211a9008ee3610e8c1e19268a38f905c32e23d"
    },
    {
      "file": "61008006_ERROR_reuse.json.gz",
      "sha256": "9735ea62b291f8635b3235b55f1330d195eb8156a610ca719ed816c25c85b6f5"
    },
    {
      "file": "61008006_P005_lifetime.json.gz",
      "sha256": "e41de272e0065b23d16bd0984f7a4e00de08fbee444c4181eb4c07a5fcc2d1b5"
    },
    {
      "file": "61008006_P005_reuse.json.gz",
      "sha256": "47dfa9e4424271795d0c9d79243cf3b937f5919b5caf58fdb80f06d538e8aa2b"
    },
    {
      "file": "61008006_P0005_lifetime.json.gz",
      "sha256": "d17fad1197c847b574dba33a7af4cfd3498724125cb21bd1f8549e3ed5dd0be1"
    },
    {
      "file": "61008006_P0005_reuse.json.gz",
      "sha256": "b24a0a02c06831363489f26632eabe7d0c8b56e123fd515e9fee209c0cc3a655"
    },
    {
      "file": "61008006_S3_CUE_lifetime.json.gz",
      "sha256": "74839b871e6a8b0ae1a8c34ed98b1742f37ec1fc3d495ba439442104fb357fec"
    },
    {
      "file": "61008006_S3_CUE_reuse.json.gz",
      "sha256": "5c707d8855c0c17e8fa26aae75c9db0812c5510dfafcfb5ee79321c16e370f5f"
    },
    {
      "file": "61008006_S3_RAND_lifetime.json.gz",
      "sha256": "0931cbc8c950d23c8633ebf2149f614cc7aa0704ad9d864c377b1930ffc3ec71"
    },
    {
      "file": "61008006_S3_RAND_reuse.json.gz",
      "sha256": "cf33f18c266dab62f216b897a55d150a7d2a82ebfada04352edef5a6433644c8"
    },
    {
      "file": "61008006_REL05_lifetime.json.gz",
      "sha256": "a0acc88a7c70dec81feb80097413507182d1104907501287b4a36f10963c6ae4"
    },
    {
      "file": "61008006_REL05_reuse.json.gz",
      "sha256": "c480978b3267f9eee58c8ae462d87bf4f1425c381fe710c29b3a4ba5769c5fbd"
    },
    {
      "file": "61008006_REL10_lifetime.json.gz",
      "sha256": "f284509e21cea08bf9fbe6a6a81f73647b30dbb287f45c4d67f9c064d0acd6bd"
    },
    {
      "file": "61008006_REL10_reuse.json.gz",
      "sha256": "3cd131c0b6566438e896045e6b3a136024adc773d3811994bdcc0b284912b027"
    },
    {
      "file": "61008006_REL20_lifetime.json.gz",
      "sha256": "c3062db5952af336ccc77a31220df66af43b6193de94527f6302bd86a077727e"
    },
    {
      "file": "61008006_REL20_reuse.json.gz",
      "sha256": "7946f1ed83ae408a865c7d446a86a149b30354653e4f4933a41294450e41926e"
    },
    {
      "file": "61008006_REL_PERM10_lifetime.json.gz",
      "sha256": "b82b9a54a8f87fcfd87a8375570a67ee0c422276df0fdcc7aa5dd3c32d0f350e"
    },
    {
      "file": "61008006_REL_PERM10_reuse.json.gz",
      "sha256": "2d11e96768461970afd4cb0d4fee74e3ce8481bf7ca908359edad07b36af2f03"
    },
    {
      "file": "61008006_FIRST10_lifetime.json.gz",
      "sha256": "c066a7a9196e20d7ba130254102e63b5b71649eb3102b33bff6ff88eb256048b"
    },
    {
      "file": "61008006_FIRST10_reuse.json.gz",
      "sha256": "e379870db6c459a5e0e0014d3c9e7111428442706fa2f157137e1e37bd49a97e"
    },
    {
      "file": "61008007_CENTER_lifetime.json.gz",
      "sha256": "316c752661894d94fa4d9cc253770743942f730a6fb622341425e5de9a07af00"
    },
    {
      "file": "61008007_CENTER_reuse.json.gz",
      "sha256": "c54f2c2859127cd1861d0ddad8508c678b70f094c1b81731364b23a1c9ec89e8"
    },
    {
      "file": "61008007_ERROR_lifetime.json.gz",
      "sha256": "75cd0c4965d03ab7aab7dc962c97dd0ca818a95c4706413540e86e8dc0324be4"
    },
    {
      "file": "61008007_ERROR_reuse.json.gz",
      "sha256": "88bf989505ca8454ccbda6411db5021b16b1fc0e20912cbbd831c6afff879a8a"
    },
    {
      "file": "61008007_P005_lifetime.json.gz",
      "sha256": "d0707f9e341430468bed12408d840fd48c29205665fe23f245372ee9f042a2a4"
    },
    {
      "file": "61008007_P005_reuse.json.gz",
      "sha256": "21c6fbbea225be8b7b22345e9d8e622213d6ce45b3ac4b22308fcb7306ce39b7"
    },
    {
      "file": "61008007_P0005_lifetime.json.gz",
      "sha256": "2db20a205ad155f5a3d0415f7ead0c0b11a6b4920e4f999c8d66808cb7aedce0"
    },
    {
      "file": "61008007_P0005_reuse.json.gz",
      "sha256": "495e21eb04f489654d820ac63dc81a827b923ac086931664f5d8070c37197d92"
    },
    {
      "file": "61008007_S3_CUE_lifetime.json.gz",
      "sha256": "82c64b7e2e0a67cf8c5ce14c43e364fe5aeb8ba9530c50c0d85ba76ade47aa13"
    },
    {
      "file": "61008007_S3_CUE_reuse.json.gz",
      "sha256": "a5f35d6c8c1baf97e7448ed05d8b404c0026b81e8b6ed7c5636c309562faf9ac"
    },
    {
      "file": "61008007_S3_RAND_lifetime.json.gz",
      "sha256": "23f23f2d5d9dcdd1de89d046339b75a789e98db873c17faa6fb398589a61381a"
    },
    {
      "file": "61008007_S3_RAND_reuse.json.gz",
      "sha256": "eb4b7802a4bf8c3507363548225ac627ab1dd452c869f42c1434edd8bad9ffab"
    },
    {
      "file": "61008007_REL05_lifetime.json.gz",
      "sha256": "61d040fe7991c53e595c85118e8f3fab888d0a1e832d3fdef1fd55cbc81c5e8c"
    },
    {
      "file": "61008007_REL05_reuse.json.gz",
      "sha256": "3c7f418d0aeefaab262bcfb3e73873032415f02197de9017021a62486bf5d0e3"
    },
    {
      "file": "61008007_REL10_lifetime.json.gz",
      "sha256": "73b684314793a1095a02b6bfcb4d8ec9af2fb3838487c30bd9b2e37d63aa3359"
    },
    {
      "file": "61008007_REL10_reuse.json.gz",
      "sha256": "87498d128d84f4d9b599e3a56dabf37dea531f355655dcee42c16f3909d59ab9"
    },
    {
      "file": "61008007_REL20_lifetime.json.gz",
      "sha256": "065ac5f37e80127207bc5990c67b5a41ffb9d24640d79300fe45e1c04d2a86b8"
    },
    {
      "file": "61008007_REL20_reuse.json.gz",
      "sha256": "a1685ffc874cf4a8e6cec734cd56ee4bc8d05064b8a93ab1de199974e5e68dc7"
    },
    {
      "file": "61008007_REL_PERM10_lifetime.json.gz",
      "sha256": "abc32bdee21fc4e849fd5d01f68741f4f78dae1e7c2f8fc675050ff56401dd52"
    },
    {
      "file": "61008007_REL_PERM10_reuse.json.gz",
      "sha256": "a39976ec4d397de5465b3c19c8f1b7bbebbdb55a5bd26aa39dff1bea8df47767"
    },
    {
      "file": "61008007_FIRST10_lifetime.json.gz",
      "sha256": "1dc8069560aa8600c82e763d0823187282ab4e272ef970a795d9450aa87f5995"
    },
    {
      "file": "61008007_FIRST10_reuse.json.gz",
      "sha256": "05b58429de8df3aab5eaa8a032366b22afddfcf91be643e6987743a92474186d"
    },
    {
      "file": "61008008_CENTER_lifetime.json.gz",
      "sha256": "138462ae85dc688cb8a196ebb9ea020d6f4204b62915d6f5d9b4a26815b33a63"
    },
    {
      "file": "61008008_CENTER_reuse.json.gz",
      "sha256": "04d7e2b20658ce2b6768de2168015b994c759be12c518b41312ff090efbbfef7"
    },
    {
      "file": "61008008_ERROR_lifetime.json.gz",
      "sha256": "53fbb32fb9888adefdea51496b862698c11ca3623a05b6c47eaaff97d6195238"
    },
    {
      "file": "61008008_ERROR_reuse.json.gz",
      "sha256": "8664637e5e2b54e5d16ea49b8aca58290cff18704a99be5280e3c418e7c2eb14"
    },
    {
      "file": "61008008_P005_lifetime.json.gz",
      "sha256": "0671fc7d724a4154573e628e503579aa1a9597944eb17bd420af8bd8464b3465"
    },
    {
      "file": "61008008_P005_reuse.json.gz",
      "sha256": "b3e839d4e664cdb81d5cba6b3d4482076e01c75a69dd8e62da3f4e9e92c926af"
    },
    {
      "file": "61008008_P0005_lifetime.json.gz",
      "sha256": "a48b3fa1596ce23a171aa7dcdbdedf870729817710c9d98028d36cba2eeb78ec"
    },
    {
      "file": "61008008_P0005_reuse.json.gz",
      "sha256": "9cc572a9404db818a7deb6ef3e35cd7a1274358354594028a35c286cdb928b62"
    },
    {
      "file": "61008008_S3_CUE_lifetime.json.gz",
      "sha256": "ea7ce801bdabf28520dfe9ecb18ea84bed01611add17226d95c7ee163acc0bb0"
    },
    {
      "file": "61008008_S3_CUE_reuse.json.gz",
      "sha256": "b56a1b0238eeba8756096e123b826e09a6cda57b2053d47595e8e8004314dfac"
    },
    {
      "file": "61008008_S3_RAND_lifetime.json.gz",
      "sha256": "5f379f62b1ccd4a5652dd2a93544d3a9d993f91e8e08284a1e510fc00035ea42"
    },
    {
      "file": "61008008_S3_RAND_reuse.json.gz",
      "sha256": "1ab598d07835bca88eb9c92be9e84f54b1c0d96989a9d3c9473eddcd7ff0e992"
    },
    {
      "file": "61008008_REL05_lifetime.json.gz",
      "sha256": "842acac817d72046c95f32983ac7e8a45a51e7f73ae0df905e55c83703df3f68"
    },
    {
      "file": "61008008_REL05_reuse.json.gz",
      "sha256": "32fca415b72cd843393bed103ff7b272d3b9f13c3b67cdc14821bd6301268b5d"
    },
    {
      "file": "61008008_REL10_lifetime.json.gz",
      "sha256": "9e0134ec15683dccb144282f24986690cf3e669c8e6d5faaec9ca578fdff0287"
    },
    {
      "file": "61008008_REL10_reuse.json.gz",
      "sha256": "30b65c94ce7e1564f81e726712d19c0d0ecbbf91fb73cb3d569cb7904599ee4c"
    },
    {
      "file": "61008008_REL20_lifetime.json.gz",
      "sha256": "7af457f1e185c54c4707180a327c13ab94f732c56361180a2ff0b2a67121f5ba"
    },
    {
      "file": "61008008_REL20_reuse.json.gz",
      "sha256": "f49f2bfc8e8c931057e8d4c62c8b92632e864346850c58c47390a5021f25a446"
    },
    {
      "file": "61008008_REL_PERM10_lifetime.json.gz",
      "sha256": "4f38e588d935b70f26906963ee39e9ecedc56a3648b6e90075af475116388aba"
    },
    {
      "file": "61008008_REL_PERM10_reuse.json.gz",
      "sha256": "87ec858ea7ffa423c277dd89694cb29066300117f79b5578be373120e38ba751"
    },
    {
      "file": "61008008_FIRST10_lifetime.json.gz",
      "sha256": "eb3722235fced23614d2ef4354b23b7e0428a525fd3679196c4d0a3833a40a81"
    },
    {
      "file": "61008008_FIRST10_reuse.json.gz",
      "sha256": "3ccd53f8c118df850231fef76c9aa9d077bb3ff98d881b368685654a87bd7f2d"
    }
  ]
}
```
