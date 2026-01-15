"""
训练日志工具
统一管理训练日志的记录和数据保存
"""

import os
import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class TrainingLogger:
    """
    训练日志记录器
    
    功能：
    1. 终端实时输出训练进度
    2. 保存训练数据到 JSON/CSV
    3. 记录训练配置和超参数
    """
    
    def __init__(self, 
                 log_dir: str,
                 algorithm_name: str,
                 config: Optional[Dict] = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            algorithm_name: 算法名称
            config: 训练配置字典
        """
        self.log_dir = log_dir
        self.algorithm_name = algorithm_name
        self.config = config or {}
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 日志文件路径
        self.json_path = os.path.join(log_dir, f'{algorithm_name}_training_data.json')
        self.csv_path = os.path.join(log_dir, f'{algorithm_name}_training_data.csv')
        self.config_path = os.path.join(log_dir, f'{algorithm_name}_config.json')
        
        # 数据存储
        self.episode_data: List[Dict] = []
        
        # 保存配置
        self._save_config()
    
    def _save_config(self):
        """保存训练配置"""
        config_info = {
            'algorithm': self.algorithm_name,
            'timestamp': self.timestamp,
            'config': self.config,
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    def log_episode(self, 
                    episode: int,
                    total_reward: float,
                    episode_length: int,
                    epsilon: float,
                    loss: Optional[float] = None,
                    extra_info: Optional[Dict] = None):
        """
        记录单个回合的数据
        
        Args:
            episode: 回合编号
            total_reward: 总奖励
            episode_length: 回合步数
            epsilon: 探索率
            loss: 损失值（可选，仅用于 DQN）
            extra_info: 额外信息字典
        """
        episode_info = {
            'episode': episode,
            'total_reward': float(total_reward),
            'episode_length': int(episode_length),
            'epsilon': float(epsilon),
            'timestamp': datetime.now().isoformat(),
        }
        
        if loss is not None:
            episode_info['loss'] = float(loss)
        
        if extra_info:
            episode_info.update(extra_info)
        
        self.episode_data.append(episode_info)
    
    def print_progress(self,
                      episode: int,
                      total_episodes: int,
                      avg_reward_100: float,
                      success_rate_100: float,
                      epsilon: float,
                      avg_loss: Optional[float] = None):
        """
        打印训练进度（终端输出）
        
        Args:
            episode: 当前回合
            total_episodes: 总回合数
            avg_reward_100: 最近100回合平均奖励
            success_rate_100: 最近100回合成功率
            epsilon: 当前探索率
            avg_loss: 平均损失（可选）
        """
        progress_str = (
            f"[{self.algorithm_name}] "
            f"回合 {episode}/{total_episodes} | "
            f"成功率: {success_rate_100:.2%} | "
            f"平均奖励: {avg_reward_100:.3f} | "
            f"探索率: {epsilon:.3f}"
        )
        
        if avg_loss is not None:
            progress_str += f" | 损失: {avg_loss:.4f}"
        
        print(progress_str)
    
    def save_json(self):
        """保存为 JSON 格式"""
        data = {
            'algorithm': self.algorithm_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'episodes': self.episode_data,
            'summary': self._generate_summary(),
        }
        
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"训练数据已保存 (JSON): {self.json_path}")
    
    def save_csv(self):
        """保存为 CSV 格式"""
        if not self.episode_data:
            print("警告: 没有数据可保存")
            return
        
        # 获取所有字段（合并所有episode的字段）
        fieldnames = set()
        for episode in self.episode_data:
            fieldnames.update(episode.keys())
        fieldnames = sorted(list(fieldnames))  # 排序以保证顺序一致
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episode_data)
        
        print(f"训练数据已保存 (CSV): {self.csv_path}")
    
    def save_all(self):
        """保存所有格式"""
        self.save_json()
        self.save_csv()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        生成训练摘要统计
        
        Returns:
            摘要字典
        """
        if not self.episode_data:
            return {}
        
        rewards = [ep['total_reward'] for ep in self.episode_data]
        lengths = [ep['episode_length'] for ep in self.episode_data]
        
        # 计算成功率（最后100回合）
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
        
        summary = {
            'total_episodes': len(self.episode_data),
            'final_success_rate_100': success_rate,
            'avg_reward': float(np.mean(rewards)),
            'avg_reward_100': float(np.mean(recent_rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'avg_episode_length': float(np.mean(lengths)),
        }
        
        # 如果有损失数据
        if 'loss' in self.episode_data[0]:
            losses = [ep['loss'] for ep in self.episode_data if 'loss' in ep]
            if losses:
                summary['avg_loss'] = float(np.mean(losses))
                summary['final_loss_100'] = float(np.mean(losses[-100:]))
        
        return summary
    
    def load_from_json(self, json_path: str) -> Dict:
        """
        从 JSON 文件加载训练数据
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            训练数据字典
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.episode_data = data.get('episodes', [])
        return data


class BenchmarkLogger:
    """
    Benchmark 实验日志管理器
    
    管理多个算法的训练日志
    """
    
    def __init__(self, base_dir: str, map_size: str, mode: str):
        """
        初始化 Benchmark 日志管理器
        
        Args:
            base_dir: 基础目录
            map_size: 地图大小 ('4x4' 或 '8x8')
            mode: 模式 ('deterministic' 或 'stochastic')
        """
        self.base_dir = base_dir
        self.map_size = map_size
        self.mode = mode
        
        # 创建实验目录
        self.experiment_dir = os.path.join(base_dir, f'{map_size}_{mode}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 算法日志记录器
        self.loggers: Dict[str, TrainingLogger] = {}
        
        # 实验元数据
        self.metadata = {
            'map_size': map_size,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'algorithms': [],
        }
    
    def create_logger(self, 
                     algorithm_name: str, 
                     config: Optional[Dict] = None) -> TrainingLogger:
        """
        为指定算法创建日志记录器
        
        Args:
            algorithm_name: 算法名称
            config: 训练配置
            
        Returns:
            训练日志记录器
        """
        logger = TrainingLogger(self.experiment_dir, algorithm_name, config)
        self.loggers[algorithm_name] = logger
        
        if algorithm_name not in self.metadata['algorithms']:
            self.metadata['algorithms'].append(algorithm_name)
        
        return logger
    
    def save_metadata(self):
        """保存实验元数据"""
        metadata_path = os.path.join(self.experiment_dir, 'experiment_metadata.json')
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"实验元数据已保存: {metadata_path}")
    
    def save_all_logs(self):
        """保存所有算法的日志"""
        for logger in self.loggers.values():
            logger.save_all()
        
        self.save_metadata()
    
    def generate_comparison_report(self, save_path: Optional[str] = None) -> str:
        """
        生成对比报告
        
        Args:
            save_path: 保存路径（可选）
            
        Returns:
            Markdown 格式报告
        """
        report_lines = [
            f"# 算法对比实验报告\n",
            f"**地图大小**: {self.map_size}",
            f"**模式**: {self.mode}",
            f"**实验时间**: {self.metadata['timestamp']}\n",
            "---\n",
            "## 训练结果对比\n",
            "| 算法 | 总回合数 | 最终成功率 | 平均奖励 | 平均步数 |",
            "|------|----------|-----------|----------|----------|",
        ]
        
        for algo_name, logger in self.loggers.items():
            summary = logger._generate_summary()
            report_lines.append(
                f"| {algo_name} | "
                f"{summary.get('total_episodes', 'N/A')} | "
                f"{summary.get('final_success_rate_100', 0):.2%} | "
                f"{summary.get('avg_reward_100', 0):.3f} | "
                f"{summary.get('avg_episode_length', 0):.1f} |"
            )
        
        report = "\n".join(report_lines)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"对比报告已保存: {save_path}")
        
        return report


# 测试代码
if __name__ == '__main__':
    # 测试单个日志记录器
    logger = TrainingLogger('test_logs', 'Q-Learning', {'lr': 0.8, 'gamma': 0.95})
    
    # 模拟训练
    for i in range(100):
        logger.log_episode(
            episode=i,
            total_reward=np.random.randn() + i * 0.01,
            episode_length=np.random.randint(10, 50),
            epsilon=1.0 - i * 0.01,
        )
    
    logger.save_all()
    
    # 测试 Benchmark 日志管理器
    benchmark_logger = BenchmarkLogger('test_results', '8x8', 'stochastic')
    
    for algo in ['Q-Learning', 'SARSA', 'DQN', 'DDQN']:
        algo_logger = benchmark_logger.create_logger(algo, {'test': True})
        for i in range(50):
            algo_logger.log_episode(i, np.random.randn(), 20, 0.5)
    
    benchmark_logger.save_all_logs()
    report = benchmark_logger.generate_comparison_report('test_results/8x8_stochastic/report.md')
    print("\n" + report)
    
    print("\n测试完成！")
