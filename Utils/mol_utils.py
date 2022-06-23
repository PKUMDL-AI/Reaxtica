import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import re


# 保存好的反应模板
rxn_templates = [
    AllChem.ReactionFromSmarts('[c;+0:1]-[F;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1].[F;+0:2]-[B;+0:4]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Cl;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1].[Cl;+0:2]-[B;+0:4]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Br;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1].[Br;+0:2]-[B;+0:4]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[I;+0:2].[c;+0:3]-[B;+0:4]>>[c;+0:3]-[c;+0:1].[I;+0:2]-[B;+0:4]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[F;+0:2].[N;+0:3]>>[c;+0:1]-[N;+0:3].[F;+0:2]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Cl;+0:2].[N;+0:3]>>[c;+0:1]-[N;+0:3].[Cl;+0:2]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[Br;+0:2].[N;+0:3]>>[c;+0:1]-[N;+0:3].[Br;+0:2]'),
    AllChem.ReactionFromSmarts('[c;+0:1]-[I;+0:2].[N;+0:3]>>[c;+0:1]-[N;+0:3].[I;+0:2]')]


def remove_mapnum(mol):  # 去除所有原子的MapNum rdkitMol→rdkitMol
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def GetAtomWithMapNumber(mol, mapid):  # 得到MapNum对应的原子 rdkitMol，int→rdkitAtom
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapid:
            return atom



def extract_substrate(rxn_sml):  # 从反应物中提取底物、试剂和产物 str→str, str, str
    # print(rxn_sml)
    post1 = rxn_sml.find('>')
    if post1 == -1: # 对于没有给出产物的情况，直接返回底物、试剂
        rxn1, rxn2 = rxn_sml.split('.')
        r1, r2 = Chem.MolFromSmiles(rxn1), Chem.MolFromSmiles(rxn2)
        for atom in r1.GetAtoms():
            if atom.GetNumRadicalElectrons() == 1:
                return rxn2, '', rxn1
        for atom in r2.GetAtoms():
            if atom.GetNumRadicalElectrons() == 1:
                return rxn1, '', rxn2
        substrate = rxn1 if len(Chem.MolFromSmiles(rxn1).GetAtoms()) > len(
            Chem.MolFromSmiles(rxn2).GetAtoms()) else rxn2
        reagent = rxn1 if (substrate == rxn2) else rxn2
        return substrate, '', reagent
    post2 = rxn_sml.rfind('>')
    product = rxn_sml[post2 + 1:]
    rxn1, rxn2 = rxn_sml[:post1].split('.')
    r1, r2, p = Chem.MolFromSmiles(rxn1), Chem.MolFromSmiles(rxn2), Chem.MolFromSmiles(product)
    virtual_product = [Chem.MolToInchi(i[j][0]) for i in
                       (rxn.RunReactants((r1, r2)) for rxn in rxn_templates) if
                       len(i) > 0 for j in range(len(i))]  # 如果有反应规则可以利用，则可以精确判断底物和产物
    if Chem.MolToInchi(Chem.MolFromSmiles(product)) in virtual_product:
        # print(1)
        return rxn1, product, rxn2
    virtual_product = [Chem.MolToInchi(i[j][0]) for i in
                       (rxn.RunReactants((r2, r1)) for rxn in rxn_templates) if
                       len(i) > 0 for j in range(len(i))]
    if Chem.MolToInchi(Chem.MolFromSmiles(product)) in virtual_product:
        # print(2)
        return rxn2, product, rxn1
    for atom in r1.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            return rxn2, product, rxn1
    for atom in r2.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            return rxn1, product, rxn2
    if p.HasSubstructMatch(r1):
        return rxn1, product, rxn2
    if p.HasSubstructMatch(r2):
        return rxn2, product, rxn1
    substrate = rxn1 if len(Chem.MolFromSmiles(rxn1).GetAtoms()) > len(Chem.MolFromSmiles(rxn2).GetAtoms()) else rxn2
    reagent = rxn1 if (substrate == rxn2) else rxn2  # 否则从反应式中简单地提取底物（反应物中原子更多的）与试剂（较少的）并返回 ⚠可能会很不准
    return substrate, product, reagent


def get_core_map(r, p):  # 对于给定的底物和产物，指出底物中的反应核心是哪些原子 str, str→set(int)
    core_mapnum = set()
    r_mol = Chem.MolFromSmiles(r)
    p_mol = Chem.MolFromSmiles(p)
    r_dict = {a.GetAtomMapNum(): a for a in r_mol.GetAtoms()}
    p_dict = {a.GetAtomMapNum(): a for a in p_mol.GetAtoms()}
    for a_map in r_dict:  # 根据编号找出反应前后发生了反应的原子
        try:
            a_neighbor_in_p = set([a.GetAtomMapNum() for a in p_dict[a_map].GetNeighbors()])
        except KeyError:
            continue  # 如果一个原子在反应中被取代了，那么这个原子将不会被列入反应核心
        a_neighbor_in_r = set([a.GetAtomMapNum() for a in r_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_r:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_r:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mol.GetBondBetweenAtoms(r_dict[a_neighbor].GetIdx(), r_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)
    return core_mapnum

def retro_DA_SMARTS(sml):
    rxn = AllChem.ReactionFromSmarts(
        '[C;+0:3]1-[C;+0:1]-[C;+0:2]-[C;+0:6]-[C;+0:5]=[C;+0:4]-1>>[C;+0:1]=[C;+0:2].[C;+0:3]=[C;+0:4]-[C;+0:5]=['
        'C;+0:6]')
    post = sml.find('>>')
    reactant1, reactant2 = sml[:post].split('.')
    r1_sml = re.sub(r'[\\\/]', '', reactant1)  # 除去反应物双键的顺反
    r2_sml = re.sub(r'[\\\/]', '', reactant2)
    r1 = Chem.MolFromSmiles(r1_sml)
    r1_inchi = Chem.MolToInchi(r1)
    r2 = Chem.MolFromSmiles(r2_sml)
    r2_inchi = Chem.MolToInchi(r2)
    # print(r1_inchi, r2_inchi)
    product = sml[post + 2:]
    p = Chem.MolFromSmiles(product)
    p_inchi = Chem.MolToInchi(Chem.MolFromSmiles(product))
    DA_product = [(i[0], i[1]) for i in rxn.RunReactants([p])]
    for DA in DA_product:
        virtual_r1_sml, virtual_r2_sml = Chem.MolToSmiles(DA[0]), Chem.MolToSmiles(DA[1])
        virtual_r1_sml = re.sub(r'[\\\/]', '', virtual_r1_sml)  # 除去虚拟产物中双键的顺反
        virtual_r2_sml = re.sub(r'[\\\/]', '', virtual_r2_sml)
        if (
                Chem.MolToInchi(Chem.MolFromSmiles(virtual_r1_sml)),
                Chem.MolToInchi(Chem.MolFromSmiles(virtual_r2_sml))) == (
                r1_inchi, r2_inchi):  # 利用Inchi进行比对可以回避MapNum造成SMILES不同的问题
            return 0  # 和SMARTS反应模板相同的顺序代表dieneophile在前
        if (
                Chem.MolToInchi(Chem.MolFromSmiles(virtual_r1_sml)),
                Chem.MolToInchi(Chem.MolFromSmiles(virtual_r2_sml))) == (
                r2_inchi, r1_inchi):
            return 1  # 相反的配对顺序代表diene在前
    return -1


def extract_substituent(sml, order):  # 利用反应模板提取取代基，回避了Mapping出错的问题
    rxn = AllChem.ReactionFromSmarts(
        '[C;+0:1]=[C;+0:2].[C;+0:3]=[C;+0:4]-[C;+0:5]=[C;+0:6]>>[C;+0:3]1-[C;+0:1]-[C;+0:2]-[C;+0:6]-[C;+0:5]=[C;+0:4]-1')
    post = sml.find('>>')
    reactant1, reactant2 = sml[:post].split('.')
    r1_sml = reactant1
    r2_sml = reactant2
    r1 = Chem.MolFromSmiles(r1_sml)
    r1_inchi = Chem.MolToInchi(r1)  # 同上，利用去除顺反异构的InChi回避MapNum引入的区别
    r2 = Chem.MolFromSmiles(r2_sml)
    r2_inchi = Chem.MolToInchi(r2)
    product = sml[post + 2:]
    p_sml = product
    p = Chem.MolFromSmiles(p_sml)
    real_product = -1
    if order == 0:
        DA_product = [i[0] for i in rxn.RunReactants((r1, r2))]
    elif order == 1:
        DA_product = [i[0] for i in rxn.RunReactants((r2, r1))]
    # print('.'.join([Chem.MolToSmiles(j) for j in DA_product]))
    for mol in DA_product:
        if Chem.MolToInchi(mol) == Chem.MolToInchi(p):
            real_product = mol  # 利用InChi筛选找到虚拟产物中的真正产物
    if real_product == -1:
        r1_sml = re.sub(r'[\\\/]', '', reactant1)
        r2_sml = re.sub(r'[\\\/]', '', reactant2)
        r1 = Chem.MolFromSmiles(r1_sml)
        r1_inchi = Chem.MolToInchi(r1)  # 同上，利用去除顺反异构的InChi回避MapNum引入的区别
        r2 = Chem.MolFromSmiles(r2_sml)
        r2_inchi = Chem.MolToInchi(r2)
        product = sml[post + 2:]
        p_sml = re.sub(r'[\\\/]', '', product)
        p = Chem.MolFromSmiles(p_sml)
        if order == 0:
            DA_product = [i[0] for i in rxn.RunReactants((r1, r2))]
        elif order == 1:
            DA_product = [i[0] for i in rxn.RunReactants((r2, r1))]
        for mol in DA_product:
            if Chem.MolToInchi(mol) == Chem.MolToInchi(p):
                real_product = mol  # 利用InChi筛选找到虚拟产物中的真正产物
    # print(Chem.MolToSmiles(real_product))
    reaction_core = []
    for atom in real_product.GetAtoms():
        if atom.GetAtomMapNum() == 0:  # 利用虚拟反应会除去反应中心MapNum这一特性，筛选出精确真正的反应中心
            reaction_core.append(atom)
    for atom in reaction_core:  # 依次给反应中心手动标号101~106，其中101-104源于diene，105-106源于dieneophile
        flag = 0
        for bond in atom.GetBonds():
            if bond.GetBondType() == 2 and bond.IsInRing():  # 注意这根双键必须在环上，否则有可能被环外双键弄乱
                atom.SetAtomMapNum(102)  # 从双键的一端开始标为102
                flag = 1
                break
        if flag:
            break
    for atom in reaction_core:
        bond102 = real_product.GetBondBetweenAtoms(atom.GetIdx(), GetAtomWithMapNumber(real_product, 102).GetIdx())
        try:
            if atom.GetAtomMapNum() == 0 and bond102.GetBondType() == 2:
                atom.SetAtomMapNum(103)  # 双键的另一侧为103
            if atom.GetAtomMapNum() == 0 and bond102.GetBondType() == 1:
                atom.SetAtomMapNum(101)  # 与第一个原子单键相连的另一个原子标为101
        except:
            pass
    for atom in reaction_core:
        bond103 = real_product.GetBondBetweenAtoms(atom.GetIdx(), GetAtomWithMapNumber(real_product, 103).GetIdx())
        bond101 = real_product.GetBondBetweenAtoms(atom.GetIdx(), GetAtomWithMapNumber(real_product, 101).GetIdx())
        if atom.GetAtomMapNum() == 0:
            try:
                if bond103.GetBondType() == 1:
                    atom.SetAtomMapNum(104)
            except:
                try:
                    if bond101.GetBondType() == 1:
                        atom.SetAtomMapNum(106)
                except:
                    atom.SetAtomMapNum(105)  # 剩下三个原子按顺序标为104-106
    # print(Chem.MolToSmiles(real_product))
    bondidx = []
    real_product = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(real_product)))  # real_product即为对产物反应中心六元环重新标号后的结构
    bondidx.append(real_product.GetBondBetweenAtoms(GetAtomWithMapNumber(real_product, 106).GetIdx(),
                                                    GetAtomWithMapNumber(real_product, 101).GetIdx()).GetIdx())
    for i in range(1, 6):
        bondidx.append(real_product.GetBondBetweenAtoms(GetAtomWithMapNumber(real_product, 100 + i).GetIdx(),
                                                        GetAtomWithMapNumber(real_product, 101 + i).GetIdx()).GetIdx())
    fragments = [i for i in Chem.MolToSmiles(Chem.FragmentOnBonds(real_product, bondidx)).split('.')]  # 第一次打碎反应中心（六元环的六根键）
    orig_subs = [[] * i for i in range(6)]
    subs = [[] * i for i in range(6)]
    for i in range(6):
        for fragment in fragments:
            if str(101 + i) in fragment:  # 在所有碎片中搜寻含有对应原子的片段，回避了环状结构带来的问题
                orig_sub = Chem.AddHs(Chem.MolFromSmiles(re.sub(r'\[[0-9]*\*\]', '[Ce]', fragment)))  #
                # 将所有反应中心中的键替换为与Ce相连
                break
        core_atom = GetAtomWithMapNumber(orig_sub, 101 + i)
        sub_bondidx = [i.GetIdx() for i in core_atom.GetBonds()]
        orig_subs[i] = [j for j in Chem.MolToSmiles(Chem.FragmentOnBonds(orig_sub, sub_bondidx)).split('.')]  #
        # 第二次打碎所有反应中心原子相连的键，得到取代基
        for sub in orig_subs[i]:
            if str(101 + i) not in sub and not (sub[-4:] == '[Ce]' and len(sub) <= 10):
                subs[i].append(sub)
    for i in (0, 3, 4, 5):
        if len(subs[i]) == 1:
            subs[i].append(subs[i][0])  # 对于应该有两个取代基的位点，如果只有一个，说明该位点为螺环，需要copy一份
    # print('.'.join(['.'.join(sub) for sub in orig_subs]))
    convert_sub = [[] * i for i in range(6)]
    for i in range(6):
        if i in (0, 3, 4, 5):
            if subs[i][0] == subs[i][1] and len(subs[i][0]) > 10:  # 把Ce替换的位置换回C，把真正相连的位置替换为La
                convert_sub[i].append(re.sub(r'\[Ce\]', 'C', re.sub(r'\[[0-9]+\*\]', 'C',
                                                                    re.sub(r'\[[0-9]+\*\]', '[La]', subs[i][0],
                                                                           count=1))))
                convert_sub[i].append(re.sub(r'\[Ce\]', 'C', re.sub(r'\[La\]', 'C',
                                                                    re.sub(r'\[[0-9]+\*\]', '[La]', subs[i][1],
                                                                           count=2), count=1)))
            else:
                convert_sub[i].append(re.sub(r'\[Ce\]', 'C', re.sub(r'\[[0-9]+\*\]', '[La]', subs[i][0])))
                convert_sub[i].append(re.sub(r'\[Ce\]', 'C', re.sub(r'\[[0-9]+\*\]', '[La]', subs[i][1])))
        else:
            convert_sub[i].append(re.sub(r'\[Ce\]', 'C', re.sub(r'\[[0-9]+\*\]', '[La]', subs[i][0])))
    allsub = '.'.join(['.'.join(convert_sub[i]) for i in range(6)])  # 将所有取代基join成为一个vector
    return Chem.MolToSmiles(real_product), allsub

