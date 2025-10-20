#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试ESM3和UniMol特征提取功能
"""
from __future__ import annotations

import torch
from pathlib import Path

from src.features.esm3_protein import ESM3ProteinEncoder, ESM3ProteinEncoderConfig
from src.features.unimol_ligand import PaiNNLigandEncoder, PaiNNLigandEncoderConfig
from src.features.ligand import load_ligand_from_file


def test_esm3_protein():
    """测试ESM3蛋白质特征提取"""
    print("开始测试ESM3蛋白质特征提取...")
    
    # 创建ESM3蛋白质编码器配置
    protein_cfg = ESM3ProteinEncoderConfig(
        model_name="esm3_sm_open_v1",  # 或其他ESM3模型名称
        device="cpu",  # 使用CPU进行测试
        fp16=False,
        chunk_size=1024,
        proj_in=1536,  # ESM3输出维度
        proj_out=512   # 投影到512维
    )
    
    try:
        encoder = ESM3ProteinEncoder(protein_cfg)
        print("ESM3蛋白质编码器创建成功")
        
        # 测试一些氨基酸残基
        residues = ["ALA", "GLY", "LEU", "VAL", "SER"]
        features = encoder.encode(residues)
        
        print(f"蛋白质特征提取成功: {features.shape}")
        print(f"特征维度: {features.shape[-1]}")
        
        return True
    except Exception as e:
        print(f"ESM3蛋白质特征提取测试失败: {e}")
        return False


def test_unimol_ligand():
    """测试UniMol小分子特征提取"""
    print("\n开始测试UniMol小分子特征提取...")
    
    # 创建UniMol配体编码器配置
    ligand_cfg = PaiNNLigandEncoderConfig(
        projection_dim=512,
        max_atoms=256,
        use_gpu=False,  # 使用CPU进行测试
    )
    
    try:
        encoder = PaiNNLigandEncoder(ligand_cfg)
        print("UniMol配体编码器创建成功")
        
        # 创建一个简单的分子用于测试
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")  # 乙醇
        if mol is None:
            print("无法创建测试分子")
            return False
            
        # 测试特征提取
        features, info = encoder.featurize(mol)
        
        print(f"小分子特征提取成功: {features.shape}")
        print(f"特征维度: {features.shape[-1]}")
        print(f"坐标信息: {info['coords'].shape}")
        print(f"键索引: {info['bond_index'].shape}")
        
        return True
    except Exception as e:
        print(f"UniMol小分子特征提取测试失败: {e}")
        return False


def test_with_real_data():
    """使用真实数据测试（如果存在）"""
    print("\n开始使用真实数据测试...")
    
    # 测试配体加载和特征提取
    try:
        # 尝试加载一个SDF文件（如果存在）
        test_sdf_path = Path("test_ligand.sdf")
        if test_sdf_path.exists():
            mol = load_ligand_from_file(str(test_sdf_path))
            print(f"成功加载配体: {Chem.MolToSmiles(mol)}")
            
            # 使用UniMol编码
            ligand_cfg = PaiNNLigandEncoderConfig(
                projection_dim=512,
                max_atoms=256,
                use_gpu=False,
            )
            encoder = PaiNNLigandEncoder(ligand_cfg)
            features, info = encoder.featurize(mol)
            
            print(f"真实配体特征提取成功: {features.shape}")
            return True
        else:
            print("未找到测试配体文件，跳过真实数据测试")
            # 使用SMILES创建一个测试分子
            from rdkit import Chem
            mol = Chem.MolFromSmiles("c1ccccc1")  # 苯
            if mol:
                print(f"使用测试分子: {Chem.MolToSmiles(mol)}")
                
                # 使用UniMol编码
                ligand_cfg = PaiNNLigandEncoderConfig(
                    projection_dim=512,
                    max_atoms=256,
                    use_gpu=False,
                )
                encoder = PaiNNLigandEncoder(ligand_cfg)
                features, info = encoder.featurize(mol)
                
                print(f"测试分子特征提取成功: {features.shape}")
                return True
    except Exception as e:
        print(f"真实数据测试失败: {e}")
        return False
    
    return False


def main():
    """主函数"""
    print("开始测试ESM3和UniMol特征提取功能")
    
    # 测试ESM3蛋白质特征提取
    esm3_success = test_esm3_protein()
    
    # 测试UniMol小分子特征提取
    unimol_success = test_unimol_ligand()
    
    # 测试真实数据
    real_data_success = test_with_real_data()
    
    print(f"\n测试结果:")
    print(f"ESM3蛋白质特征提取: {'成功' if esm3_success else '失败'}")
    print(f"UniMol小分子特征提取: {'成功' if unimol_success else '失败'}")
    print(f"真实数据测试: {'成功' if real_data_success else '失败'}")
    
    overall_success = esm3_success and unimol_success
    print(f"整体功能测试: {'成功' if overall_success else '失败'}")
    
    return overall_success


if __name__ == "__main__":
    main()