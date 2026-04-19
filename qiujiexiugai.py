#qiujie代码的修改版本
#kimi版本，能够初步求解，后续还需要继续深化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import math
import os
from typing import List, Dict, Tuple, Optional
import copy

# 复用您现有的类定义
from build_initial import (
    InitialScenarioBuilder, EnemyTarget, ReconNode, CommNode, 
    DecisionNode, AttackNode, AssessNode, BaseNode
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KillChainSolver(InitialScenarioBuilder):
    """
    杀伤链动态构建与协同优化求解器
    继承InitialScenarioBuilder以复用数据加载和基础可视化功能
    """
    
    def __init__(self, data_folder: str = "primary_situation"):
        super().__init__(data_folder)
        self.current_time = 0          # 当前仿真时间
        self.time_step = 10            # 时间步长：10秒
        self.visualization_interval = 50  # 可视化间隔：50秒
        self.kill_chains = {}          # 当前时间步的杀伤链方案 {target_id: chain_dict}
        self.solution_history = []     # 历史方案记录
        
        # 算法参数
        self.light_speed = 299792.458  # 光速 km/s
        self.jamming_alpha = 0.15      # 毁伤概率衰减系数
        self.weight_damage = 0.5       # 毁伤效能权重
        self.weight_delay = 0.5        # 时延权重  
        self.ongoing_attacks = []  # 记录进行中的打击任务
        self.target_attack_history = {}
        # 新增：目标打击历史记录 {target_id: [{'time': int, 'chain': str, 'ammo': int, 'hit_time': float, 'P_kill': float}, ...]}
        #self.target_attack_history = {}
    def update_target_positions(self, delta_time: float = None):
        """
        动态更新敌方目标坐标（匀速直线运动）
        
        Args:
            delta_time: 时间增量(秒)，默认为self.time_step
        """
        if delta_time is None:
            delta_time = self.time_step
            
        for target in self.targets:
            if target.status == 0:  # 已被摧毁的目标不再移动
                continue
                
            # 将速度m/s转换为km/时间步: speed_mps * delta_time / 1000
            distance_km = target.speed_mps * delta_time / 1000
            heading_rad = math.radians(target.heading_deg)
            
            # 计算位移
            dx = distance_km * math.cos(heading_rad)
            dy = distance_km * math.sin(heading_rad)
            
            # 更新位置
            new_x = target.position[0] + dx
            new_y = target.position[1] + dy
            target.position = (new_x, new_y)
            
            # 更新剩余飞临时间
            target.time_to_target_s -= delta_time
            
            # 检查是否到达保卫要地（距离原点小于5km）
            dist_to_center = math.sqrt(new_x**2 + new_y**2)
            if dist_to_center <= self.protected_area['radius']:
                target.status = -1  # -1表示已突破防御
                print(f"警告: {target.node_id} 已突破保卫要地！")
    
    def get_jamming_intensity(self, x: float, y: float) -> float:
        """计算指定位置的电子干扰强度（取最大值）"""
        max_jamming = 0
        for zone in self.jamming_zones:
            dist = math.sqrt((x - zone['center_x'])**2 + (y - zone['center_y'])**2)
            if dist <= zone['radius']:
                # 简化为区域内均匀干扰，实际可改为距离衰减模型
                max_jamming = max(max_jamming, zone['intensity'])
        return max_jamming
    
    def check_node_capacity(self, node: BaseNode) -> bool:
        """检查节点是否还有剩余容量"""
        return node.status == 1 and node.current_tasks < node.capacity
    
    def calculate_link_delay(self, from_node, to_node, link_type: str, jamming: float = 0) -> float:
        """
        计算链路时延
        
        Args:
            link_type: 'TO'(侦察), 'OC'(通信), 'CD'(决策), 'DA'(打击), 'AE'(评估), 'ED'(反馈)
        """
        if link_type in ['TO', 'OC', 'AE', 'ED']:
            # 传输时延 + 处理/发送时延
            dist = math.sqrt((from_node.position[0] - to_node.position[0])**2 + 
                           (from_node.position[1] - to_node.position[1])**2)
            trans_delay = dist / self.light_speed
            
            if link_type == 'TO':
                # 侦察边：传输+处理时延
                return trans_delay + to_node.processing_delay_s
            elif link_type == 'OC':
                # 通信边：传输+发送时延（考虑带宽和干扰）
                bandwidth_eff = max(0.2, 1 - jamming * 0.8)  # 干扰导致带宽效率下降
                send_delay = 1.0 / (to_node.bandwidth_mbps * bandwidth_eff)  # 假设传输1M数据
                return trans_delay + send_delay
            elif link_type == 'AE':
                return trans_delay + to_node.processing_delay_s
                # 评估/反馈边
            elif link_type == 'ED':
                return trans_delay + from_node.processing_delay_s
                
                
        elif link_type == 'CD':
            # 决策边：主要是决策时延
            return to_node.decision_delay_s
            
        elif link_type == 'DA':
            # 打击边：准备时间 + 导弹飞行时间
            dist = math.sqrt((from_node.position[0] - to_node.position[0])**2 + 
                           (from_node.position[1] - to_node.position[1])**2)
            fly_time = dist * 1000 / to_node.missile_speed_mps  # km转m，除以m/s得秒
            return to_node.preparation_time_s + fly_time
        
        return 0
    
    def build_feasible_chains(self, target: EnemyTarget) -> List[Dict]:
        """
        为单个目标构建所有可行杀伤链（O-C-D-A-E-D）
        
        Returns:
            可行链列表，每条链包含节点组合、时延、毁伤概率等指标
        """
        feasible_chains = []
        tx, ty = target.position
        target_jamming = self.get_jamming_intensity(tx, ty)
        
        # 1. 筛选可用侦察节点（探测范围内且有容量）
        available_O = []
        for o in self.recon_nodes:
            if not self.check_node_capacity(o):
                continue
            dist = math.sqrt((o.position[0] - tx)**2 + (o.position[1] - ty)**2)
            if dist <= o.detection_radius_km:
                # 计算信噪比或探测质量（简化）
                detection_quality = o.anti_jamming_coeff * (1 - target_jamming * 0.5)
                if detection_quality > 0.3:  # 阈值
                    available_O.append((o, dist))
        
        # 2. 筛选可用打击节点（射程内且有弹药和容量）
        available_A = []
        for a in self.attack_nodes:
            if not self.check_node_capacity(a) or a.current_ammunition <= 0:
                continue
            dist = math.sqrt((a.position[0] - tx)**2 + (a.position[1] - ty)**2)
            if dist <= a.range_km:
                available_A.append((a, dist))
        
        # 3. 筛选可用评估节点
        available_E = []
        for e in self.assess_nodes:
            if not self.check_node_capacity(e):
                continue
            dist = math.sqrt((e.position[0] - tx)**2 + (e.position[1] - ty)**2)
            if dist <= e.assessment_radius_km:
                available_E.append((e, dist))
        
        # 4. 遍历组合构建完整链
        for o, dist_OT in available_O:
            for c in self.comm_nodes:
                if not self.check_node_capacity(c):
                    continue
                # 检查O-C连通性（在通信半径内）
                dist_OC = math.sqrt((o.position[0] - c.position[0])**2 + 
                                   (o.position[1] - c.position[1])**2)
                if dist_OC > c.communication_radius_km:
                    continue
                
                for d in self.decision_nodes:
                    if not self.check_node_capacity(d):
                        continue
                    # 检查C-D连通性
                    dist_CD = math.sqrt((c.position[0] - d.position[0])**2 + 
                                       (c.position[1] - d.position[1])**2)
                    if dist_CD > c.communication_radius_km:
                        continue
                    
                    for a, dist_AT in available_A:
                        # 检查D-A连通性（指令传输）
                        dist_DA = math.sqrt((d.position[0] - a.position[0])**2 + 
                                           (d.position[1] - a.position[1])**2)
                        if dist_DA > c.communication_radius_km:  # 复用C的通信半径作为通用标准
                            continue
                        
                        for e, dist_ET in available_E:
                            # 检查E-D反馈链路
                            dist_ED = math.sqrt((e.position[0] - d.position[0])**2 + 
                                               (e.position[1] - d.position[1])**2)
                            if dist_ED > c.communication_radius_km:
                                continue
                            
                            # 构建杀伤链
                            chain = {
                                'target': target,
                                'nodes': {
                                    'O': o, 'C': c, 'D': d, 'A': a, 'E': e
                                },
                                'distances': {
                                    'OT': dist_OT, 'OC': dist_OC, 'CD': dist_CD,
                                    'AT': dist_AT, 'DA': dist_DA, 'ED': dist_ED
                                }
                            }
                            
                            # 计算性能指标
                            metrics = self.evaluate_chain(chain, target_jamming)
                            chain.update(metrics)
                            
                            # 检查时间约束：总时延 < 剩余飞临时间
                            if chain['total_delay'] < target.time_to_target_s * 0.9:  # 留10%余量
                                feasible_chains.append(chain)
        
        return feasible_chains
    
    def evaluate_chain(self, chain: Dict, jamming: float) -> Dict:
        """
        计算杀伤链的时延和毁伤概率
        """
        nodes = chain['nodes']
        o, c, d, a, e = nodes['O'], nodes['C'], nodes['D'], nodes['A'], nodes['E']
        
        # 分段计算时延
        tau_TO = self.calculate_link_delay(chain['target'], o, 'TO')
        tau_OC = self.calculate_link_delay(o, c, 'OC', jamming)
        tau_CD = self.calculate_link_delay(c, d, 'CD')
        tau_DA = self.calculate_link_delay(d, a, 'DA')
        tau_AE = self.calculate_link_delay(a, e, 'AE')
        tau_ED = self.calculate_link_delay(e, d, 'ED')
        
        total_delay = tau_TO + tau_OC + tau_CD + tau_DA + tau_AE + tau_ED
        
        # 毁伤概率计算（指数衰减模型）
        P_base = a.single_hit_prob
        P_kill = P_base * math.exp(-self.jamming_alpha * jamming * (1 - o.anti_jamming_coeff))
        P_kill = max(0.1, min(0.95, P_kill))  # 边界处理
        
        return {
            'total_delay': total_delay,
            'P_kill': P_kill,
            'stages_delay': {
                'TO': tau_TO, 'OC': tau_OC, 'CD': tau_CD,
                'DA': tau_DA, 'AE': tau_AE, 'ED': tau_ED
            }
        }
    
    def optimize_kill_chains(self) -> Dict[str, Dict]:
        """
        多目标优化：为所有目标分配最优杀伤链组合
        使用贪心算法+权重分配（可扩展为整数规划或遗传算法）
        """
        # 为每个目标生成可行链
        target_chains = {}
        for target in self.targets:
            if target.status == 1:  # 只考虑存活且未突破的目标
                chains = self.build_feasible_chains(target)
                if chains:
                    target_chains[target.node_id] = chains
        
        if not target_chains:
            return {}
        
        # 按威胁值排序目标（高威胁优先）
        sorted_targets = sorted(
            target_chains.items(),
            key=lambda x: next(t.threat_value for t in self.targets if t.node_id == x[0]),
            reverse=True
        )
        
        # 贪心分配（考虑资源冲突）
        solution = {}
        node_loads = {}  # 记录节点负载 {node_id: current_load}
        
        # 初始化负载
        for node in (self.recon_nodes + self.comm_nodes + self.decision_nodes + 
                    self.attack_nodes + self.assess_nodes):
            node_loads[node.node_id] = 0
        
        for target_id, chains in sorted_targets:
            best_chain = None
            best_score = -float('inf')
            
            for chain in chains:
                nodes = chain['nodes']
                
                # 检查资源约束
                feasible = True
                for node_type, node in nodes.items():
                    if node_loads[node.node_id] >= node.capacity:
                        feasible = False
                        break
                    if node_type == 'A' and node.current_ammunition <= 0:
                        feasible = False
                        break

                if not feasible:
                    continue
                
                # 计算综合得分（归一化后的加权和）
                # 毁伤概率越大越好，时延越小越好
                norm_damage = chain['P_kill']  # 已在0-1之间
                norm_delay = 1 - min(chain['total_delay'] / 300, 1)  # 假设最大时延300s
                
                score = (self.weight_damage * norm_damage + 
                        self.weight_delay * norm_delay)
                
                # 威胁值加权（高威胁目标优先分配资源）
                target = chain['target']
                score *= (1 + target.threat_value / 10)
                
                if score > best_score:
                    best_score = score
                    best_chain = chain
            
            if best_chain:
                solution[target_id] = best_chain
                
                # 更新节点负载
                for node_type, node in best_chain['nodes'].items():
                    node_loads[node.node_id] += 1
                    node.current_tasks += 1
                
                # 消耗弹药
                #best_chain['nodes']['A'].current_ammunition -= 1
        
        return solution
    
    def visualize_current_situation(self, filename: str = None):
        """
        可视化当前态势（增强版，显示杀伤链链路）
        """
        if filename is None:
            filename = f"{self.current_time}_s_situation.png"
        save_path = os.path.join(self.data_folder, filename)
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # 标题和坐标设置
        ax.set_title(f'典型场景下的杀伤链动态构建与协同优化 - t={self.current_time}s', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X坐标 (km)', fontsize=14)
        ax.set_ylabel('Y坐标 (km)', fontsize=14)
        ax.set_xlim(-self.battlefield_width/2 - 10, self.battlefield_width/2 + 10)
        ax.set_ylim(-self.battlefield_height/2 - 10, self.battlefield_height/2 + 10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # 绘制保卫要地
        protected = Circle((0, 0), self.protected_area['radius'], 
                          color='gold', alpha=0.3, edgecolor='red', linewidth=3)
        ax.add_patch(protected)
        ax.text(0, 0, '保卫要地', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='darkred')
        
        # 绘制干扰区域
        for zone in self.jamming_zones:
            jam_circle = Circle((zone['center_x'], zone['center_y']), 
                               zone['radius'], color='purple', alpha=0.15)
            ax.add_patch(jam_circle)
            ax.text(zone['center_x'], zone['center_y'], 
                   f'干扰\n{zone["intensity"]:.1f}', ha='center', va='center', 
                   fontsize=10, color='purple', alpha=0.7)
        
        # 节点样式定义
        node_style = {
            'O': {'color': 'blue', 'marker': 's', 'size': 250, 'label': '侦察(O)'},
            'C': {'color': 'green', 'marker': 'o', 'size': 250, 'label': '通信(C)'},
            'D': {'color': 'purple', 'marker': 'D', 'size': 250, 'label': '决策(D)'},
            'A': {'color': 'darkorange', 'marker': '*', 'size': 400, 'label': '打击(A)'},
            'E': {'color': 'cyan', 'marker': 'v', 'size': 250, 'label': '评估(E)'}
        }
        
        # 绘制所有节点
        all_nodes = (self.recon_nodes + self.comm_nodes + self.decision_nodes + 
                    self.attack_nodes + self.assess_nodes)
        
        for node in all_nodes:
            if node.status == 0:
                continue
            style = node_style.get(node.node_type, {})
            
            # 根据负载调整透明度
            alpha = 1.0 if node.current_tasks < node.capacity else 0.4
            
            ax.scatter(node.position[0], node.position[1], 
                      c=style.get('color'), marker=style.get('marker'),
                      s=style.get('size'), edgecolors='black', linewidth=2, 
                      alpha=alpha, zorder=5)
            
            # 节点标注
            label_text = node.node_id
            if node.node_type == 'A':
                label_text += f"\n剩{node.current_ammunition}弹"
            elif node.current_tasks > 0:
                label_text += f"\n({node.current_tasks}/{node.capacity})"
            
            ax.annotate(label_text, xy=node.position, xytext=(0, 18),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 绘制目标及杀伤链
        for target in self.targets:
            if target.status == 0:  # 已摧毁
                # 绘制X标记
                ax.scatter(target.position[0], target.position[1], 
                          c='gray', marker='x', s=200, linewidths=3, zorder=6)
                continue
            elif target.status == -1:  # 已突破
                ax.scatter(target.position[0], target.position[1], 
                          c='black', marker='s', s=300, zorder=6)
                continue
            
            # 绘制存活目标
            threat_colors = plt.cm.Reds(target.threat_value / 10)
            ax.scatter(target.position[0], target.position[1], 
                      c=[threat_colors], marker='^', s=400, 
                      edgecolors='black', linewidth=2, zorder=6)
            
            # 目标标注
            info_text = f"{target.node_id}\nT={target.threat_value}\n剩余{target.time_to_target_s:.0f}s"
            ax.annotate(info_text, xy=target.position, xytext=(0, -35),
                       textcoords='offset points', ha='center', va='top',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
            
            # 绘制航向矢量
            heading_rad = math.radians(target.heading_deg)
            arrow_len = min(12, target.speed_mps * 0.1)  # 速度越快箭头越长
            ax.arrow(target.position[0], target.position[1],
                    arrow_len * math.cos(heading_rad), 
                    arrow_len * math.sin(heading_rad),
                    head_width=3, head_length=4, fc='red', ec='red', alpha=0.7)
            
            # 绘制分配的杀伤链
            if target.node_id in self.kill_chains:
                chain = self.kill_chains[target.node_id]
                nodes = chain['nodes']
                
                # 链路路径点
                waypoints = [
                    target.position,      # T
                    nodes['O'].position,  # O
                    nodes['C'].position,  # C
                    nodes['D'].position,  # D
                    nodes['A'].position,  # A
                    nodes['E'].position,  # E
                    nodes['D'].position   # 回D
                ]
                
                # 绘制链路（不同颜色表示不同阶段）
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                for i in range(len(waypoints)-1):
                    x1, y1 = waypoints[i]
                    x2, y2 = waypoints[i+1]
                    
                    # 绘制连线
                    ax.plot([x1, x2], [y1, y2], '--', color=colors[i % len(colors)], 
                           alpha=0.6, linewidth=2, zorder=4)
                    
                    # 绘制流向箭头（中点）
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
                    ax.arrow(mid_x - dx, mid_y - dy, dx, dy,
                            head_width=2, head_length=2.5, 
                            fc=colors[i % len(colors)], 
                            ec=colors[i % len(colors)], alpha=0.8, zorder=4)
                
                # 在目标旁边显示链的性能指标
                metrics_text = f"P={chain['P_kill']:.2f}\nτ={chain['total_delay']:.1f}s"
                ax.text(target.position[0] + 8, target.position[1] + 5, metrics_text,
                       fontsize=9, color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=12, label='敌方目标', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=10, label='侦察节点(O)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='通信节点(C)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', 
                      markersize=10, label='决策节点(D)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='darkorange', 
                      markersize=15, label='打击节点(A)', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='cyan', 
                      markersize=10, label='评估节点(E)', markeredgecolor='black'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        os.makedirs(self.data_folder, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"态势图已保存: {save_path}")
        plt.close()
    
    def print_kill_chain_plan(self):
        """打印当前杀伤链方案详情"""
        print(f"\n{'='*80}")
        print(f"当前时间: {self.current_time}s 的杀伤链分配方案")
        print(f"{'='*80}")
        
        if not self.kill_chains:
            print("当前无可行杀伤链方案")
            return
        
        print(f"{'目标':<8} {'杀伤链路径':<30} {'毁伤概率':<10} {'总时延(s)':<10} {'状态'}")
        print(f"{'-'*80}")
        
        for target_id, chain in self.kill_chains.items():
            target = chain['target']
            nodes = chain['nodes']
            
            path = f"{nodes['O'].node_id}->{nodes['C'].node_id}->{nodes['D'].node_id}->" \
                   f"{nodes['A'].node_id}->{nodes['E'].node_id}"
            
            status = "已分配"
            if target.status == 0:
                status = "已摧毁"
            elif target.status == -1:
                status = "已突破"
            
            print(f"{target_id:<8} {path:<30} {chain['P_kill']:<10.3f} "
                  f"{chain['total_delay']:<10.2f} {status}")
            
            # 详细时延分解
            stages = chain['stages_delay']
            print(f"         时延分解: TO={stages['TO']:.2f}s, OC={stages['OC']:.2f}s, "
                  f"CD={stages['CD']:.2f}s, DA={stages['DA']:.2f}s, "
                  f"AE={stages['AE']:.2f}s, ED={stages['ED']:.2f}s")
        print(f"{'='*80}\n")
    
    def run_dynamic_simulation(self, max_time: float = 400):
        """
        运行动态仿真（主循环）- 支持跨周期打击跟踪与文件输出优化
        
        Args:
            max_time: 最大仿真时间（秒）
        """
        print("启动杀伤链动态构建与协同优化仿真...")
        print(f"时间步长: {self.time_step}s, 可视化间隔: {self.visualization_interval}s")
        
        # 加载初始数据（复用父类方法）
        self.build_initial_scenario()
        
        # 初始化进行中的打击任务队列
        self.ongoing_attacks = []
        
        # 创建simulate子文件夹用于保存结果
        simulate_dir = os.path.join(self.data_folder, "simulate")
        os.makedirs(simulate_dir, exist_ok=True)
        
        # 初始化方案日志文件（记录杀伤链分配详情）
        log_file_path = os.path.join(simulate_dir, "kill_chain_plan.txt")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write(f"杀伤链动态构建与协同优化方案记录\n")
        log_file.write(f"仿真参数: 时间步长={self.time_step}s, 最大时间={max_time}s, 权重(毁伤/时延)={self.weight_damage}/{self.weight_delay}\n")
        log_file.write(f"{'='*100}\n\n")
        
        try:
            # 保存初始状态（t=0）
            self.current_time = 0
            self.kill_chains = {}  
            # 保存初始图片到simulate文件夹
            self.visualize_current_situation(os.path.join("simulate", "0_s_situation.png"))
            log_file.write(f"[时间: 0s] 初始态势已保存至 simulate/0_s_situation.png\n\n")
            
            # 主循环
            while self.current_time < max_time:
                self.current_time += self.time_step
                
                print(f"\n[仿真时间: {self.current_time}s]")
                log_file.write(f"\n{'='*100}\n")
                log_file.write(f"[仿真时间: {self.current_time}s]\n")
                log_file.write(f"{'-'*100}\n")
                
                # 1. 更新目标位置
                self.update_target_positions()
                
                # 2. 处理进行中的打击任务（检查导弹是否到达目标）
                completed_attacks = []
                for attack in self.ongoing_attacks:
                    if self.current_time >= attack['hit_time']:
                        target = next((t for t in self.targets if t.node_id == attack['target_id']), None)
                        if target and target.status == 1:  # 目标仍存活且未突破
                            if np.random.random() < attack['P_kill']:
                                target.status = 0
                                msg = f"[打击效果] {attack['target_id']} 被 {attack['attacker_id']} 摧毁 (命中时间: {self.current_time}s)"
                                print(f"  {msg}")
                                log_file.write(f"{msg}\n")
                            else:
                                msg = f"[打击效果] {attack['target_id']} 被 {attack['attacker_id']} 打击但未摧毁 (脱靶)"
                                print(f"  {msg}")
                                log_file.write(f"{msg}\n")
                        completed_attacks.append(attack)
                
                # 移除已完成的任务
                for attack in completed_attacks:
                    self.ongoing_attacks.remove(attack)
                
                # 3. 重置节点任务计数（动态重规划策略）
                for node in (self.recon_nodes + self.comm_nodes + self.decision_nodes + 
                            self.attack_nodes + self.assess_nodes):
                    node.current_tasks = 0
                
                # 4. 重新优化杀伤链（基于当前态势）
                self.kill_chains = self.optimize_kill_chains()
                current_plan_entries = [] 
                # 5. 分配新的打击任务到队列（避免重复分配）
                for target_id, chain in self.kill_chains.items():
                    target = chain['target']
                    # 检查是否已有针对该目标的进行中打击
                    already_attacking = any(a['target_id'] == target_id for a in self.ongoing_attacks)
                    
                    if not already_attacking and target.status == 1:
                        attacker_node = chain['nodes']['A']
                        
                        # 【新增】检查弹药是否充足（防御性编程）
                        if attacker_node.current_ammunition <= 0:
                            print(f"  [警告] {attacker_node.node_id} 弹药不足，跳过目标 {target_id}")
                            continue
                        
                        # 【新增】实际扣减弹药（真正发射导弹）
                        attacker_node.current_ammunition -= 1
                        
                        hit_time = self.current_time + chain['total_delay']
                        
                        # 记录到历史
                        if target_id not in self.target_attack_history:
                            self.target_attack_history[target_id] = []
                        
                        history_entry = {
                            'assign_time': self.current_time,
                            'hit_time': hit_time,
                            'chain_path': f"{chain['nodes']['O'].node_id}->{chain['nodes']['C'].node_id}->{chain['nodes']['D'].node_id}->{attacker_node.node_id}->{chain['nodes']['E'].node_id}",
                            'attacker': attacker_node.node_id,
                            'ammo': 1,  # 实际消耗的弹药数
                            'P_kill': chain['P_kill'],
                            'total_delay': chain['total_delay']
                        }
                        self.target_attack_history[target_id].append(history_entry)
                        
                        self.ongoing_attacks.append({
                            'target_id': target_id,
                            'hit_time': hit_time,
                            'P_kill': chain['P_kill'],
                            'attacker_id': attacker_node.node_id,
                            'launch_time': self.current_time
                        })
                        print(f"  [任务分配] {target_id} -> {attacker_node.node_id}, "
                            f"预计 {hit_time:.1f}s 命中 (时延 {chain['total_delay']:.1f}s)")
                        #print(f"  {msg}")
                        #log_file.write(f"{msg}\n")
                    
                    # 构建当前方案记录（用于txt输出）
                    nodes = chain['nodes']
                    path_str = f"{nodes['O'].node_id}->{nodes['C'].node_id}->{nodes['D'].node_id}->{nodes['A'].node_id}->{nodes['E'].node_id}"
                    
                    # 确定状态
                    status_str = "已分配"
                    if target.status == 0:
                        status_str = "已摧毁"
                    elif target.status == -1:
                        status_str = "已突破"
                    
                    # 记录弹药分配（当前模型每个目标分配1发，从对应A节点扣除）
                    #ammo_assigned = 1
                    
                    entry = {
                        'target_id': target_id,
                        'path': path_str,
                        'attacker': nodes['A'].node_id,
                        'ammo': 1,
                        'P_kill': chain['P_kill'],
                        'total_delay': chain['total_delay'],
                        'status': status_str,
                        'stages': chain['stages_delay']
                    }
                    current_plan_entries.append(entry)
                
                # 6. 输出当前方案到日志文件（包含详细时延分解）
                if current_plan_entries:
                    log_file.write(f"\n当前杀伤链分配方案详情:\n")
                    log_file.write(f"{'目标':<10} {'杀伤链路径(O-C-D-A-E)':<35} {'攻击节点':<10} {'弹药数':<8} {'毁伤概率':<10} {'总时延(s)':<12} {'状态'}\n")
                    log_file.write(f"{'-'*100}\n")
                    
                    for entry in current_plan_entries:
                        log_file.write(f"{entry['target_id']:<10} {entry['path']:<35} "
                                    f"{entry['attacker']:<10} {entry['ammo']:<8} "
                                    f"{entry['P_kill']:<10.3f} {entry['total_delay']:<12.2f} "
                                    f"{entry['status']}\n")
                        # 详细时延分解
                        s = entry['stages']
                        log_file.write(f"          └─ 时延分解: TO={s['TO']:.2f}s | OC={s['OC']:.2f}s | "
                                    f"CD={s['CD']:.2f}s | DA={s['DA']:.2f}s | "
                                    f"AE={s['AE']:.2f}s | ED={s['ED']:.2f}s\n")
                else:
                    log_file.write("当前无可行杀伤链方案\n")
                
                # 显示进行中任务状态
                if self.ongoing_attacks:
                    msg = f"[进行中任务] 共有 {len(self.ongoing_attacks)} 个导弹正在飞行"
                    print(f"  {msg}")
                    log_file.write(f"\n{msg}:\n")
                    for atk in self.ongoing_attacks:
                        remaining = atk['hit_time'] - self.current_time
                        log_file.write(f"  - {atk['target_id']}: 由{atk['attacker_id']}发射, "
                                    f"预计{remaining:.1f}s后命中 (毁伤概率:{atk['P_kill']:.2f})\n")
                
                log_file.flush()  # 确保实时写入磁盘
                
                # 7. 可视化（每50秒）- 保存到simulate子文件夹
                if self.current_time % self.visualization_interval == 0:
                    viz_filename = os.path.join("simulate", f"{self.current_time}_s_situation.png")
                    self.visualize_current_situation(viz_filename)
                    log_file.write(f"\n[可视化] 态势图已保存至 {viz_filename}\n")
                
                # 8. 检查终止条件（所有目标已摧毁或突破，且无进行中任务）
                active_targets = [t for t in self.targets if t.status == 1]
                if not active_targets and not self.ongoing_attacks:
                    print("\n所有目标已处理完毕，仿真结束")
                    log_file.write(f"\n{'='*100}\n")
                    log_file.write("仿真结束原因: 所有目标已摧毁或突破，且无进行中打击任务\n")
                    break
            
            # 最终可视化（如果最后一步不是50的整数倍）
            if self.current_time % self.visualization_interval != 0:
                viz_filename = os.path.join("simulate", f"{self.current_time}_s_situation.png")
                self.visualize_current_situation(viz_filename)
            
            # 写入最终统计报告到日志
            log_file.write(f"\n{'='*100}\n")
            log_file.write("最终统计报告\n")
            log_file.write(f"{'-'*100}\n")
            
            destroyed = len([t for t in self.targets if t.status == 0])
            breached = len([t for t in self.targets if t.status == -1])
            remaining = len([t for t in self.targets if t.status == 1])
            
            log_file.write(f"总仿真时间: {self.current_time}s\n")
            log_file.write(f"目标统计: 摧毁={destroyed} | 突破={breached} | 剩余={remaining} | 总计={len(self.targets)}\n")
            
            # 各打击节点弹药消耗详情
            log_file.write(f"\n弹药消耗详情:\n")
            for a in self.attack_nodes:
                used = a.ammunition - a.current_ammunition
                remaining_ammo = a.current_ammunition
                log_file.write(f"  {a.node_id}: 初始={a.ammunition} | 消耗={used} | 剩余={remaining_ammo}\n")
            
            log_file.write(f"{'='*100}\n")
                
        finally:
            # 生成总体方案汇总
            log_file.write(f"\n\n{'='*100}\n")
            log_file.write("【总体方案汇总表】\n")
            log_file.write(f"{'='*100}\n")
            
            if not self.target_attack_history:
                log_file.write("无打击任务记录\n")
            else:
                # 表头
                log_file.write(f"{'目标ID':<10} {'分配次数':<10} {'总弹药消耗':<12} {'最终状态':<10} {'详细打击记录'}\n")
                log_file.write(f"{'-'*100}\n")
                
                # 按目标ID排序遍历
                for target_id in sorted(self.target_attack_history.keys()):
                    attacks = self.target_attack_history[target_id]
                    total_ammo = sum(a['ammo'] for a in attacks)
                    attack_count = len(attacks)
                    
                    # 获取目标最终状态
                    target_obj = next((t for t in self.targets if t.node_id == target_id), None)
                    if target_obj:
                        if target_obj.status == 0:
                            final_status = "已摧毁"
                        elif target_obj.status == -1:
                            final_status = "已突破"
                        else:
                            final_status = "存活/未处理"
                    else:
                        final_status = "未知"
                    
                    # 构建详细记录字符串（每次打击的链路和弹药）
                    details = []
                    for i, atk in enumerate(attacks, 1):
                        details.append(f"[{i}]t={atk['assign_time']}s:{atk['chain_path']}(P={atk['P_kill']:.2f},弹={atk['ammo']})")
                    
                    detail_str = " | ".join(details)
                    
                    log_file.write(f"{target_id:<10} {attack_count:<10} {total_ammo:<12} {final_status:<10} {detail_str}\n")
                
                # 各打击节点弹药消耗汇总（与之前保持一致，但格式优化）
                log_file.write(f"\n{'-'*100}\n")
                log_file.write("【弹药消耗统计-by 节点】\n")
                for a in self.attack_nodes:
                    used = a.ammunition - a.current_ammunition
                    log_file.write(f"  {a.node_id}: 初始={a.ammunition}发 | 消耗={used}发 | 剩余={a.current_ammunition}发\n")
            
            log_file.write(f"{'='*100}\n")
            log_file.close()
            print(f"\n方案日志已保存至: {log_file_path}")
        
        # 控制台输出最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终统计报告"""
        print(f"\n{'='*80}")
        print("杀伤链动态构建仿真最终报告")
        print(f"{'='*80}")
        
        destroyed = len([t for t in self.targets if t.status == 0])
        breached = len([t for t in self.targets if t.status == -1])
        remaining = len([t for t in self.targets if t.status == 1])
        
        print(f"总仿真时间: {self.current_time}s")
        print(f"摧毁目标数: {destroyed}/{len(self.targets)}")
        print(f"突破防御数: {breached}/{len(self.targets)}")
        print(f"剩余目标数: {remaining}/{len(self.targets)}")
        
        # 资源消耗统计
        print(f"\n资源消耗统计:")
        for a in self.attack_nodes:
            used = a.ammunition - a.current_ammunition
            print(f"  {a.node_id}: 消耗弹药 {used}/{a.ammunition}")
        
        print(f"{'='*80}")


def main():
    """主函数"""
    # 创建求解器实例
    solver = KillChainSolver(data_folder="primary_situation")
    
    # 运行动态仿真（最大1000秒）
    solver.run_dynamic_simulation(max_time=1000)


if __name__ == "__main__":
    main()