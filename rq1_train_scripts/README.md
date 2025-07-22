### 1. Monolingual Fine-Tuning (Supervised)

# 1. Make each executable: 

## SALT
```

chmod +x train_scripts/train_salt_afroxlmr.sh

chmod +x train_scripts/train_salt_mbert.sh

chmod +x train_scripts/train_salt_xlmr.sh

```

## MPTC
```

chmod +x train_scripts/train_mptc_afroxlmr.sh

chmod +x train_scripts/train_mptc_mbert.sh

chmod +x train_scripts/train_mptc_xlmr.sh

```

## COMBINED
```

chmod +x train_scripts/train_combined_afroxlmr.sh

chmod +x train_scripts/train_combined_mbert.sh

chmod +x train_scripts/train_combined_xlmr.sh

```

# 2. Run each:

## SALT
```

./train_scripts/train_salt_afroxlmr.sh

./train_scripts/train_salt_mbert.sh

./train_scripts/train_salt_xlmr.sh

```

## MPTC
```

./train_scripts/train_mptc_afroxlmr.sh

./train_scripts/train_mptc_mbert.sh

./train_scripts/train_mptc_xlmr.sh

```

## COMBINED
```

./train_scripts/train_combined_afroxlmr.sh

./train_scripts/train_combined_mbert.sh

./train_scripts/train_combined_xlmr.sh

```

### 2. Zero-Shot Transfer from Auxiliary Languages

## SALT

# 1 bam
```
chmod +x train_scripts/ZeroShotTransfer/bam/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bam/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bam/SALT/train_salt_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/ZeroShotTransfer/fon/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/fon/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/fon/SALT/train_salt_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/ZeroShotTransfer/hau/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/hau/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/hau/SALT/train_salt_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/ZeroShotTransfer/kin/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/kin/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/kin/SALT/train_salt_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/ZeroShotTransfer/lug/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/lug/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/lug/SALT/train_salt_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/ZeroShotTransfer/luo/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/luo/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/luo/SALT/train_salt_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/ZeroShotTransfer/mos/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/mos/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/mos/SALT/train_salt_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/ZeroShotTransfer/nya/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/nya/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/nya/SALT/train_salt_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/ZeroShotTransfer/sna/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/sna/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/sna/SALT/train_salt_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/ZeroShotTransfer/swa/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/swa/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/swa/SALT/train_salt_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/ZeroShotTransfer/twi/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/twi/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/twi/SALT/train_salt_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/ZeroShotTransfer/wol/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/wol/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/wol/SALT/train_salt_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/ZeroShotTransfer/xho/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/xho/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/xho/SALT/train_salt_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/ZeroShotTransfer/yor/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/yor/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/yor/SALT/train_salt_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/ZeroShotTransfer/zul/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/zul/SALT/train_salt_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/zul/SALT/train_salt_xlmr.sh
```

## MPTC

# 1 bam
```
chmod +x train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_xlmr.sh
```

## COMBINED

# 1 bam
```
chmod +x train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_xlmr.sh
```

### 3. Cross-lingual + Runyankore Fine-Tuning

## SALT

# 1 bam
```
chmod +x train_scripts/CrossLingualCombined/bam/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bam/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bam/SALT/train_salt_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/CrossLingualCombined/bbj/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bbj/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bbj/SALT/train_salt_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/CrossLingualCombined/ewe/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ewe/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ewe/SALT/train_salt_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/CrossLingualCombined/fon/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/fon/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/fon/SALT/train_salt_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/CrossLingualCombined/hau/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/hau/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/hau/SALT/train_salt_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/CrossLingualCombined/ibo/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ibo/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ibo/SALT/train_salt_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/CrossLingualCombined/kin/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/kin/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/kin/SALT/train_salt_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/CrossLingualCombined/lug/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/lug/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/lug/SALT/train_salt_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/CrossLingualCombined/luo/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/luo/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/luo/SALT/train_salt_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/CrossLingualCombined/mos/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/mos/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/mos/SALT/train_salt_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/CrossLingualCombined/nya/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/nya/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/nya/SALT/train_salt_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/CrossLingualCombined/pcm/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/pcm/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/pcm/SALT/train_salt_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/CrossLingualCombined/sna/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/sna/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/sna/SALT/train_salt_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/CrossLingualCombined/swa/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/swa/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/swa/SALT/train_salt_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/CrossLingualCombined/tsn/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/tsn/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/tsn/SALT/train_salt_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/CrossLingualCombined/twi/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/twi/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/twi/SALT/train_salt_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/CrossLingualCombined/wol/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/wol/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/wol/SALT/train_salt_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/CrossLingualCombined/xho/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/xho/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/xho/SALT/train_salt_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/CrossLingualCombined/yor/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/yor/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/yor/SALT/train_salt_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/CrossLingualCombined/zul/SALT/train_salt_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/zul/SALT/train_salt_mbert.sh

chmod +x train_scripts/CrossLingualCombined/zul/SALT/train_salt_xlmr.sh
```

## MPTC

# 1 bam
```
chmod +x train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_mbert.sh

chmod +x train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_xlmr.sh
```

## COMBINED

# 1 bam
```
chmod +x train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_xlmr.sh
```

# 2 bbj
```
chmod +x train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_xlmr.sh
```

# 3 ewe
```
chmod +x train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_xlmr.sh
```

# 4 fon
```
chmod +x train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_xlmr.sh
```

# 5 hau
```
chmod +x train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_xlmr.sh
```

# 6 ibo
```
chmod +x train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_xlmr.sh
```

# 7 kin
```
chmod +x train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_xlmr.sh
```

# 8 lug
```
chmod +x train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_xlmr.sh
```

# 9 luo
```
chmod +x train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_xlmr.sh
```

# 10 mos
```
chmod +x train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_xlmr.sh
```

# 11 nya
```
chmod +x train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_xlmr.sh
```

# 12 pcm
```
chmod +x train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_xlmr.sh
```

# 13 sna
```
chmod +x train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_xlmr.sh
```

# 14 swa
```
chmod +x train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_xlmr.sh
```

# 15 tsn
```
chmod +x train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_xlmr.sh
```

# 16 twi
```
chmod +x train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_xlmr.sh
```

# 17 wol
```
chmod +x train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_xlmr.sh
```

# 18 xho
```
chmod +x train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_xlmr.sh
```

# 19 yor
```
chmod +x train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_xlmr.sh
```

# 20 zul
```
chmod +x train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_afroxlmr.sh

chmod +x train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_mbert.sh

chmod +x train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_xlmr.sh
```

-----------------------------------------------------------------------------------------------------------

# RUN EACH:

### 2. Zero-Shot Transfer from Auxiliary Languages

## SALT

# 1 bam
```
./train_scripts/ZeroShotTransfer/bam/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bam/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/bam/SALT/train_salt_xlmr.sh
```

# 2 bbj
```
./train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/bbj/SALT/train_salt_xlmr.sh
```

# 3 ewe
```
./train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/ewe/SALT/train_salt_xlmr.sh
```

# 4 fon
```
./train_scripts/ZeroShotTransfer/fon/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/fon/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/fon/SALT/train_salt_xlmr.sh
```

# 5 hau
```
./train_scripts/ZeroShotTransfer/hau/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/hau/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/hau/SALT/train_salt_xlmr.sh
```

# 6 ibo
```
./train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/ibo/SALT/train_salt_xlmr.sh
```

# 7 kin
```
./train_scripts/ZeroShotTransfer/kin/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/kin/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/kin/SALT/train_salt_xlmr.sh
```

# 8 lug
```
./train_scripts/ZeroShotTransfer/lug/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/lug/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/lug/SALT/train_salt_xlmr.sh
```

# 9 luo
```
./train_scripts/ZeroShotTransfer/luo/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/luo/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/luo/SALT/train_salt_xlmr.sh
```

# 10 mos
```
./train_scripts/ZeroShotTransfer/mos/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/mos/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/mos/SALT/train_salt_xlmr.sh
```

# 11 nya
```
./train_scripts/ZeroShotTransfer/nya/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/nya/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/nya/SALT/train_salt_xlmr.sh
```

# 12 pcm
```
./train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/pcm/SALT/train_salt_xlmr.sh
```

# 13 sna
```
./train_scripts/ZeroShotTransfer/sna/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/sna/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/sna/SALT/train_salt_xlmr.sh
```

# 14 swa
```
./train_scripts/ZeroShotTransfer/swa/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/swa/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/swa/SALT/train_salt_xlmr.sh
```

# 15 tsn
```
./train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/tsn/SALT/train_salt_xlmr.sh
```

# 16 twi
```
./train_scripts/ZeroShotTransfer/twi/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/twi/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/twi/SALT/train_salt_xlmr.sh
```

# 17 wol
```
./train_scripts/ZeroShotTransfer/wol/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/wol/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/wol/SALT/train_salt_xlmr.sh
```

# 18 xho
```
./train_scripts/ZeroShotTransfer/xho/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/xho/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/xho/SALT/train_salt_xlmr.sh
```

# 19 yor
```
./train_scripts/ZeroShotTransfer/yor/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/yor/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/yor/SALT/train_salt_xlmr.sh
```

# 20 zul
```
./train_scripts/ZeroShotTransfer/zul/SALT/train_salt_afroxlmr.sh

./train_scripts/ZeroShotTransfer/zul/SALT/train_salt_mbert.sh

./train_scripts/ZeroShotTransfer/zul/SALT/train_salt_xlmr.sh
```

## MPTC

# 1 bam
```
./train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/bam/MPTC/train_mptc_xlmr.sh
```

# 2 bbj
```
./train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/bbj/MPTC/train_mptc_xlmr.sh
```

# 3 ewe
```
./train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/ewe/MPTC/train_mptc_xlmr.sh
```

# 4 fon
```
./train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/fon/MPTC/train_mptc_xlmr.sh
```

# 5 hau
```
./train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/hau/MPTC/train_mptc_xlmr.sh
```

# 6 ibo
```
./train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/ibo/MPTC/train_mptc_xlmr.sh
```

# 7 kin
```
./train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/kin/MPTC/train_mptc_xlmr.sh
```

# 8 lug
```
./train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/lug/MPTC/train_mptc_xlmr.sh
```

# 9 luo
```
./train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/luo/MPTC/train_mptc_xlmr.sh
```

# 10 mos
```
./train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/mos/MPTC/train_mptc_xlmr.sh
```

# 11 nya
```
./train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/nya/MPTC/train_mptc_xlmr.sh
```

# 12 pcm
```
./train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/pcm/MPTC/train_mptc_xlmr.sh
```

# 13 sna
```
./train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/sna/MPTC/train_mptc_xlmr.sh
```

# 14 swa
```
./train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/swa/MPTC/train_mptc_xlmr.sh
```

# 15 tsn
```
./train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/tsn/MPTC/train_mptc_xlmr.sh
```

# 16 twi
```
./train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/twi/MPTC/train_mptc_xlmr.sh
```

# 17 wol
```
./train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/wol/MPTC/train_mptc_xlmr.sh
```

# 18 xho
```
./train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/xho/MPTC/train_mptc_xlmr.sh
```

# 19 yor
```
./train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/yor/MPTC/train_mptc_xlmr.sh
```

# 20 zul
```
./train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_afroxlmr.sh

./train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_mbert.sh

./train_scripts/ZeroShotTransfer/zul/MPTC/train_mptc_xlmr.sh
```

## COMBINED

# 1 bam
```
./train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/bam/COMBINED/train_combined_xlmr.sh
```

# 2 bbj
```
./train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/bbj/COMBINED/train_combined_xlmr.sh
```

# 3 ewe
```
./train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/ewe/COMBINED/train_combined_xlmr.sh
```

# 4 fon
```
./train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/fon/COMBINED/train_combined_xlmr.sh
```

# 5 hau
```
./train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/hau/COMBINED/train_combined_xlmr.sh
```

# 6 ibo
```
./train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/ibo/COMBINED/train_combined_xlmr.sh
```

# 7 kin
```
./train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/kin/COMBINED/train_combined_xlmr.sh
```

# 8 lug
```
./train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/lug/COMBINED/train_combined_xlmr.sh
```

# 9 luo
```
./train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/luo/COMBINED/train_combined_xlmr.sh
```

# 10 mos
```
./train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/mos/COMBINED/train_combined_xlmr.sh
```

# 11 nya
```
./train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/nya/COMBINED/train_combined_xlmr.sh
```

# 12 pcm
```
./train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/pcm/COMBINED/train_combined_xlmr.sh
```

# 13 sna
```
./train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/sna/COMBINED/train_combined_xlmr.sh
```

# 14 swa
```
./train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/swa/COMBINED/train_combined_xlmr.sh
```

# 15 tsn
```
./train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/tsn/COMBINED/train_combined_xlmr.sh
```

# 16 twi
```
./train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/twi/COMBINED/train_combined_xlmr.sh
```

# 17 wol
```
./train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/wol/COMBINED/train_combined_xlmr.sh
```

# 18 xho
```
./train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/xho/COMBINED/train_combined_xlmr.sh
```

# 19 yor
```
./train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/yor/COMBINED/train_combined_xlmr.sh
```

# 20 zul
```
./train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_afroxlmr.sh

./train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_mbert.sh

./train_scripts/ZeroShotTransfer/zul/COMBINED/train_combined_xlmr.sh
```

### 3. Cross-lingual + Runyankore Fine-Tuning

## SALT

# 1 bam
```
./train_scripts/CrossLingualCombined/bam/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/bam/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/bam/SALT/train_salt_xlmr.sh
```

# 2 bbj
```
./train_scripts/CrossLingualCombined/bbj/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/bbj/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/bbj/SALT/train_salt_xlmr.sh
```

# 3 ewe
```
./train_scripts/CrossLingualCombined/ewe/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/ewe/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/ewe/SALT/train_salt_xlmr.sh
```

# 4 fon
```
./train_scripts/CrossLingualCombined/fon/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/fon/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/fon/SALT/train_salt_xlmr.sh
```

# 5 hau
```
./train_scripts/CrossLingualCombined/hau/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/hau/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/hau/SALT/train_salt_xlmr.sh
```

# 6 ibo
```
./train_scripts/CrossLingualCombined/ibo/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/ibo/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/ibo/SALT/train_salt_xlmr.sh
```

# 7 kin
```
./train_scripts/CrossLingualCombined/kin/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/kin/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/kin/SALT/train_salt_xlmr.sh
```

# 8 lug
```
./train_scripts/CrossLingualCombined/lug/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/lug/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/lug/SALT/train_salt_xlmr.sh
```

# 9 luo
```
./train_scripts/CrossLingualCombined/luo/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/luo/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/luo/SALT/train_salt_xlmr.sh
```

# 10 mos
```
./train_scripts/CrossLingualCombined/mos/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/mos/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/mos/SALT/train_salt_xlmr.sh
```

# 11 nya
```
./train_scripts/CrossLingualCombined/nya/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/nya/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/nya/SALT/train_salt_xlmr.sh
```

# 12 pcm
```
./train_scripts/CrossLingualCombined/pcm/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/pcm/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/pcm/SALT/train_salt_xlmr.sh
```

# 13 sna
```
./train_scripts/CrossLingualCombined/sna/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/sna/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/sna/SALT/train_salt_xlmr.sh
```

# 14 swa
```
./train_scripts/CrossLingualCombined/swa/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/swa/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/swa/SALT/train_salt_xlmr.sh
```

# 15 tsn
```
./train_scripts/CrossLingualCombined/tsn/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/tsn/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/tsn/SALT/train_salt_xlmr.sh
```

# 16 twi
```
./train_scripts/CrossLingualCombined/twi/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/twi/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/twi/SALT/train_salt_xlmr.sh
```

# 17 wol
```
./train_scripts/CrossLingualCombined/wol/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/wol/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/wol/SALT/train_salt_xlmr.sh
```

# 18 xho
```
./train_scripts/CrossLingualCombined/xho/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/xho/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/xho/SALT/train_salt_xlmr.sh
```

# 19 yor
```
./train_scripts/CrossLingualCombined/yor/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/yor/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/yor/SALT/train_salt_xlmr.sh
```

# 20 zul
```
./train_scripts/CrossLingualCombined/zul/SALT/train_salt_afroxlmr.sh

./train_scripts/CrossLingualCombined/zul/SALT/train_salt_mbert.sh

./train_scripts/CrossLingualCombined/zul/SALT/train_salt_xlmr.sh
```

## MPTC

# 1 bam
```
./train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/bam/MPTC/train_mptc_xlmr.sh
```

# 2 bbj
```
./train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/bbj/MPTC/train_mptc_xlmr.sh
```

# 3 ewe
```
./train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/ewe/MPTC/train_mptc_xlmr.sh
```

# 4 fon
```
./train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/fon/MPTC/train_mptc_xlmr.sh
```

# 5 hau
```
./train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/hau/MPTC/train_mptc_xlmr.sh
```

# 6 ibo
```
./train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/ibo/MPTC/train_mptc_xlmr.sh
```

# 7 kin
```
./train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/kin/MPTC/train_mptc_xlmr.sh
```

# 8 lug
```
./train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/lug/MPTC/train_mptc_xlmr.sh
```

# 9 luo
```
./train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/luo/MPTC/train_mptc_xlmr.sh
```

# 10 mos
```
./train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/mos/MPTC/train_mptc_xlmr.sh
```

# 11 nya
```
./train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/nya/MPTC/train_mptc_xlmr.sh
```

# 12 pcm
```
./train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/pcm/MPTC/train_mptc_xlmr.sh
```

# 13 sna
```
./train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/sna/MPTC/train_mptc_xlmr.sh
```

# 14 swa
```
./train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/swa/MPTC/train_mptc_xlmr.sh
```

# 15 tsn
```
./train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/tsn/MPTC/train_mptc_xlmr.sh
```

# 16 twi
```
./train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/twi/MPTC/train_mptc_xlmr.sh
```

# 17 wol
```
./train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/wol/MPTC/train_mptc_xlmr.sh
```

# 18 xho
```
./train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/xho/MPTC/train_mptc_xlmr.sh
```

# 19 yor
```
./train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/yor/MPTC/train_mptc_xlmr.sh
```

# 20 zul
```
./train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_afroxlmr.sh

./train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_mbert.sh

./train_scripts/CrossLingualCombined/zul/MPTC/train_mptc_xlmr.sh
```

## COMBINED

# 1 bam
```
./train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/bam/COMBINED/train_combined_xlmr.sh
```

# 2 bbj
```
./train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/bbj/COMBINED/train_combined_xlmr.sh
```

# 3 ewe
```
./train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/ewe/COMBINED/train_combined_xlmr.sh
```

# 4 fon
```
./train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/fon/COMBINED/train_combined_xlmr.sh
```

# 5 hau
```
./train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/hau/COMBINED/train_combined_xlmr.sh
```

# 6 ibo
```
./train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/ibo/COMBINED/train_combined_xlmr.sh
```

# 7 kin
```
./train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/kin/COMBINED/train_combined_xlmr.sh
```

# 8 lug
```
./train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/lug/COMBINED/train_combined_xlmr.sh
```

# 9 luo
```
./train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/luo/COMBINED/train_combined_xlmr.sh
```

# 10 mos
```
./train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/mos/COMBINED/train_combined_xlmr.sh
```

# 11 nya
```
./train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/nya/COMBINED/train_combined_xlmr.sh
```

# 12 pcm
```
./train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/pcm/COMBINED/train_combined_xlmr.sh
```

# 13 sna
```
./train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/sna/COMBINED/train_combined_xlmr.sh
```

# 14 swa
```
./train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/swa/COMBINED/train_combined_xlmr.sh
```

# 15 tsn
```
./train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/tsn/COMBINED/train_combined_xlmr.sh
```

# 16 twi
```
./train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/twi/COMBINED/train_combined_xlmr.sh
```

# 17 wol
```
./train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/wol/COMBINED/train_combined_xlmr.sh
```

# 18 xho
```
./train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/xho/COMBINED/train_combined_xlmr.sh
```

# 19 yor
```
./train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/yor/COMBINED/train_combined_xlmr.sh
```

# 20 zul
```
./train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_afroxlmr.sh

./train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_mbert.sh

./train_scripts/CrossLingualCombined/zul/COMBINED/train_combined_xlmr.sh
```
