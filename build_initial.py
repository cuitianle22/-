import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import os
from typing import List, Dict, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BaseNode:
    """节点基类"""
    def __init__(self, node_id: str, x: float, y: float, status: int = 1):
        self.node_id = node_id  # 节点ID
        self.position = (x, y)  # 位置 (x, y) km
        self.status = status  # 工作状态: 1-正常, 0-失效
        self.current_tasks = 0  # 当前任务数量
        self.capacity = 1  # 容量限制

class EnemyTarget(BaseNode):
    """敌方目标节点类"""
    def __init__(self, target_id: str, x: float, y: float, 
                 speed: float, heading: float, threat_value: float, time_to_target: float):
        super().__init__(target_id, x, y, status=1)
        self.speed_mps = speed  # 速度 (m/s)
        self.heading_deg = heading  # 航向 (度)
        self.threat_value = threat_value  # 威胁值 (1-10)
        self.time_to_target_s = time_to_target  # 飞临时间 (s)
        
    def __repr__(self):
        return f"EnemyTarget({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"threat={self.threat_value}, TTT={self.time_to_target_s:.1f}s)"

class ReconNode(BaseNode):
    """侦察节点类 (O)"""
    def __init__(self, node_id: str, x: float, y: float, 
                 detection_radius: float, location_error: float, 
                 anti_jamming_coeff: float, processing_delay: float, capacity: int = 1):
        super().__init__(node_id, x, y, status=1)
        self.node_type = 'O'
        self.detection_radius_km = detection_radius  # 探测半径 (km)
        self.location_error_m = location_error  # 定位误差 (m)
        self.anti_jamming_coeff = anti_jamming_coeff  # 抗干扰系数 (0-1)
        self.processing_delay_s = processing_delay  # 处理时延 (s)
        
        self.capacity = capacity  # 容量限制
        
    def __repr__(self):
        return f"ReconNode({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"radius={self.detection_radius_km:.1f}km)"

class CommNode(BaseNode):
    """通信节点类 (C)"""
    def __init__(self, node_id: str, x: float, y: float, 
                 communication_radius: float, bandwidth: float, capacity: int = 1):
        super().__init__(node_id, x, y, status=1)
        self.node_type = 'C'
        self.communication_radius_km = communication_radius  # 通信半径 (km)
        self.bandwidth_mbps = bandwidth  # 带宽 (Mbps)
        self.capacity = capacity  # 容量限制
        
    def __repr__(self):
        return f"CommNode({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"radius={self.communication_radius_km:.1f}km)"

class DecisionNode(BaseNode):
    """决策节点类 (D)"""
    def __init__(self, node_id: str, x: float, y: float, 
                 decision_delay: float, capacity: int = 1):
        super().__init__(node_id, x, y, status=1)
        self.node_type = 'D'
        self.decision_delay_s = decision_delay  # 决策时延 (s)
        self.capacity = capacity  # 容量限制
        
    def __repr__(self):
        return f"DecisionNode({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"delay={self.decision_delay_s:.2f}s)"

class AttackNode(BaseNode):
    """打击节点类 (A)"""
    def __init__(self, node_id: str, x: float, y: float, 
                 range_km: float, circular_error: float, preparation_time: float,
                 ammunition: int, missile_speed: float, single_hit_prob: float, capacity: int = 1):
        super().__init__(node_id, x, y, status=1)
        self.node_type = 'A'
        self.range_km = range_km  # 射程 (km)
        self.circular_error_m = circular_error  # 圆概率误差 (m)
        self.preparation_time_s = preparation_time  # 准备时间 (s)
        self.ammunition = ammunition  # 弹药量
        self.current_ammunition = ammunition  # 当前剩余弹药量
        self.missile_speed_mps = missile_speed  # 导弹平均速度 (m/s)
        self.single_hit_prob = single_hit_prob  # 单发命中毁伤概率
        self.capacity = capacity  # 容量限制
        
    def __repr__(self):
        return f"AttackNode({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"range={self.range_km:.1f}km, ammo={self.ammunition})"

class AssessNode(BaseNode):
    """评估节点类 (E)"""
    def __init__(self, node_id: str, x: float, y: float, 
                 assessment_radius: float, processing_delay: float, capacity: int = 1):
        super().__init__(node_id, x, y, status=1)
        self.node_type = 'E'
        self.assessment_radius_km = assessment_radius  # 评估半径 (km)
        self.processing_delay_s = processing_delay  # 处理时延 (s)
        self.capacity = capacity  # 容量限制
        
    def __repr__(self):
        return f"AssessNode({self.node_id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}), " \
               f"radius={self.assessment_radius_km:.1f}km)"

class InitialScenarioBuilder:
    """初始场景构建器"""
    
    def __init__(self, data_folder: str = "primary_situation"):
        self.data_folder = data_folder
        self.targets: List[EnemyTarget] = []
        self.recon_nodes: List[ReconNode] = []
        self.comm_nodes: List[CommNode] = []
        self.decision_nodes: List[DecisionNode] = []
        self.attack_nodes: List[AttackNode] = []
        self.assess_nodes: List[AssessNode] = []
        
        # 战场参数
        self.battlefield_width = 180  # km
        self.battlefield_height = 180  # km
        self.protected_area = {'x': 0, 'y': 0, 'radius': 5}  # 保卫要地
        
        # 干扰区域（根据PDF中的描述）
        self.jamming_zones = [
            {'center_x': -30, 'center_y': 20, 'radius': 15, 'intensity': 0},
            {'center_x': 10, 'center_y': -25, 'radius': 12, 'intensity': 0},
            {'center_x': 20, 'center_y': 30, 'radius': 10, 'intensity': 0}
        ]
    
    def load_targets(self) -> List[EnemyTarget]:
        """读取敌方目标数据并构建EnemyTarget对象"""
        filepath = os.path.join(self.data_folder, "enemy_targets.xlsx")
        
        try:
            # 读取CSV文件，跳过第一行（说明行）
            df = pd.read_excel(filepath, skiprows=1)
            
            # 确保列名正确
            expected_columns = ['target_id', 'position_x_km', 'position_y_km', 
                               'speed_mps', 'heading_deg', 'threat_value', 'time_to_target_s']
            
            # 重命名列（确保顺序正确）
            df.columns = expected_columns
            
            self.targets = []
            for _, row in df.iterrows():
                target = EnemyTarget(
                    target_id=str(row['target_id']),
                    x=float(row['position_x_km']),
                    y=float(row['position_y_km']),
                    speed=float(row['speed_mps']),
                    heading=float(row['heading_deg']),
                    threat_value=float(row['threat_value']),
                    time_to_target=float(row['time_to_target_s'])
                )
                self.targets.append(target)
            
            print(f"成功加载 {len(self.targets)} 个敌方目标")
            return self.targets
            
        except Exception as e:
            print(f"加载敌方目标数据时出错: {e}")
            return []
    
    def load_friendly_nodes(self) -> Dict[str, List[BaseNode]]:
        """读取我方节点数据并构建各类节点对象"""
        filepath = os.path.join(self.data_folder, "friendly_resources.xlsx")
        
        try:
            # 读取CSV文件，跳过第一行（说明行）
            df = pd.read_excel(filepath, skiprows=1)
            
            # 重命名列（根据实际数据）
            # 注意：列名可能需要根据实际文件调整
            df.columns = ['node_id', 'node_type', 'position_x_km', 'position_y_km', 'status',
                         'jamming_intensity', 'detection_radius_km', 'location_error_m',
                         'anti_jamming_coeff', 'processing_delay_s', 'communication_radius_km',
                         'bandwidth_mbps', 'decision_delay_s', 'range_km', 'circular_error_m',
                         'preparation_time_s', 'ammunition', 'missile_speed_mps',
                         'single_hit_prob', 'assessment_radius_km', 'capacity']
            
            # 重置列表
            self.recon_nodes = []
            self.comm_nodes = []
            self.decision_nodes = []
            self.attack_nodes = []
            self.assess_nodes = []
            
            for _, row in df.iterrows():
                node_type = str(row['node_type']).strip()
                node_id = str(row['node_id'])
                x = float(row['position_x_km'])
                y = float(row['position_y_km'])
                status = int(row['status'])
                capacity = int(row['capacity']) if not pd.isna(row['capacity']) else 1
                
                if node_type == 'O':  # 侦察节点
                    node = ReconNode(
                        node_id=node_id,
                        x=x, y=y,
                        detection_radius=float(row['detection_radius_km']),
                        location_error=float(row['location_error_m']),
                        anti_jamming_coeff=float(row['anti_jamming_coeff']),
                        processing_delay=float(row['processing_delay_s']),
                        capacity=capacity
                    )
                    self.recon_nodes.append(node)
                    
                elif node_type == 'C':  # 通信节点
                    node = CommNode(
                        node_id=node_id,
                        x=x, y=y,
                        communication_radius=float(row['communication_radius_km']),
                        bandwidth=float(row['bandwidth_mbps']),
                        capacity=capacity
                    )
                    self.comm_nodes.append(node)
                    
                elif node_type == 'D':  # 决策节点
                    node = DecisionNode(
                        node_id=node_id,
                        x=x, y=y,
                        decision_delay=float(row['decision_delay_s']),
                        capacity=capacity
                    )
                    self.decision_nodes.append(node)
                    
                elif node_type == 'A':  # 打击节点
                    node = AttackNode(
                        node_id=node_id,
                        x=x, y=y,
                        range_km=float(row['range_km']),
                        circular_error=float(row['circular_error_m']),
                        preparation_time=float(row['preparation_time_s']),
                        ammunition=int(row['ammunition']),
                        missile_speed=float(row['missile_speed_mps']),
                        single_hit_prob=float(row['single_hit_prob']),
                        capacity=capacity
                    )
                    self.attack_nodes.append(node)
                    
                elif node_type == 'E':  # 评估节点
                    node = AssessNode(
                        node_id=node_id,
                        x=x, y=y,
                        assessment_radius=float(row['assessment_radius_km']),
                        processing_delay=float(row['processing_delay_s']),
                        capacity=capacity
                    )
                    self.assess_nodes.append(node)
            
            print(f"成功加载我方节点:")
            print(f"  侦察节点(O): {len(self.recon_nodes)} 个")
            print(f"  通信节点(C): {len(self.comm_nodes)} 个")
            print(f"  决策节点(D): {len(self.decision_nodes)} 个")
            print(f"  打击节点(A): {len(self.attack_nodes)} 个")
            print(f"  评估节点(E): {len(self.assess_nodes)} 个")
            
            # 返回按类型分类的节点字典
            return {
                'O': self.recon_nodes,
                'C': self.comm_nodes,
                'D': self.decision_nodes,
                'A': self.attack_nodes,
                'E': self.assess_nodes
            }
            
        except Exception as e:
            print(f"加载我方节点数据时出错: {e}")
            return {}
    
    def plot_initial_distribution(self, save_path: str = None):
        """绘制初始节点分布图"""
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 设置标题和坐标轴
        ax.set_title('典型场景下的杀伤链动态构建和协同优化研究 - 初始战场态势', 
                    fontsize=18, fontweight='bold')
        ax.set_xlabel('X坐标 (km)', fontsize=12)
        ax.set_ylabel('Y坐标 (km)', fontsize=12)
        
        # 设置坐标轴范围
        ax.set_xlim(-self.battlefield_width/2, self.battlefield_width/2)
        ax.set_ylim(-self.battlefield_height/2, self.battlefield_height/2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # 绘制保卫要地
        protected_circle = Circle((self.protected_area['x'], self.protected_area['y']),
                                 self.protected_area['radius'], 
                                 color='yellow', alpha=0.5, edgecolor='red', linewidth=2)
        ax.add_patch(protected_circle)
        ax.text(self.protected_area['x'], self.protected_area['y'], '保卫要地', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # 绘制干扰区域
        for zone in self.jamming_zones:
            jamming_circle = Circle((zone['center_x'], zone['center_y']), zone['radius'],
                                   color='purple', alpha=0.2*zone['intensity'])
            ax.add_patch(jamming_circle)
            ax.text(zone['center_x'], zone['center_y'], f'干扰{zone["intensity"]:.1f}',
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 绘制敌方目标（三角形，红色系）
        if self.targets:
            target_x = [t.position[0] for t in self.targets]
            target_y = [t.position[1] for t in self.targets]
            threat_values = [t.threat_value for t in self.targets]
            
            # 根据威胁值设置颜色深浅
            scatter_targets = ax.scatter(target_x, target_y, c=threat_values, 
                                        cmap='Reds', marker='^', s=250, 
                                        edgecolors='black', linewidth=2, zorder=5, 
                                        label='敌方目标')
            
            # 添加目标ID和威胁值标签
            for target in self.targets:
                ax.annotate(f"{target.node_id}\nT{target.threat_value}",
                           xy=(target.position[0], target.position[1]),
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', va='top', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                # 绘制目标航向箭头
                heading_rad = math.radians(target.heading_deg)
                arrow_length = 10
                dx = arrow_length * math.cos(heading_rad)
                dy = arrow_length * math.sin(heading_rad)
                ax.arrow(target.position[0], target.position[1], 
                        dx, dy, head_width=3, head_length=4, 
                        fc='red', ec='red', alpha=0.7, linewidth=1.5)
        
        # 定义节点颜色和标记（与datacreate.py保持一致）
        node_colors = {'O': 'blue', 'C': 'green', 'D': 'purple', 'A': 'orange', 'E': 'cyan'}
        node_markers = {'O': 's', 'C': 'o', 'D': 'D', 'A': '*', 'E': 'v'}
        node_labels = {'O': '侦察节点', 'C': '通信节点', 'D': '决策节点', 
                      'A': '打击节点', 'E': '评估节点'}
        
        # 绘制侦察节点 (O) - 蓝色正方形
        if self.recon_nodes:
            recon_x = [node.position[0] for node in self.recon_nodes]
            recon_y = [node.position[1] for node in self.recon_nodes]
            ax.scatter(recon_x, recon_y, color=node_colors['O'], marker=node_markers['O'],
                      s=200, label=node_labels['O'], edgecolors='black', linewidth=2, zorder=5)
            
            # 添加侦察节点ID标签和探测范围
            for node in self.recon_nodes:
                ax.annotate(node.node_id,
                           xy=(node.position[0], node.position[1]),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # 绘制探测范围（虚线圆）
                """
                detection_circle = Circle((node.position[0], node.position[1]),
                                        node.detection_radius_km,
                                         color='blue', alpha=0.1, linestyle='--', linewidth=1)
                ax.add_patch(detection_circle)
                """
        
        # 绘制通信节点 (C) - 绿色圆形
        if self.comm_nodes:
            comm_x = [node.position[0] for node in self.comm_nodes]
            comm_y = [node.position[1] for node in self.comm_nodes]
            ax.scatter(comm_x, comm_y, color=node_colors['C'], marker=node_markers['C'],
                      s=200, label=node_labels['C'], edgecolors='black', linewidth=2, zorder=5)
            
            for node in self.comm_nodes:
                ax.annotate(node.node_id,
                           xy=(node.position[0], node.position[1]),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 绘制决策节点 (D) - 紫色菱形
        if self.decision_nodes:
            decision_x = [node.position[0] for node in self.decision_nodes]
            decision_y = [node.position[1] for node in self.decision_nodes]
            ax.scatter(decision_x, decision_y, color=node_colors['D'], marker=node_markers['D'],
                      s=200, label=node_labels['D'], edgecolors='black', linewidth=2, zorder=5)
            
            for node in self.decision_nodes:
                ax.annotate(node.node_id,
                           xy=(node.position[0], node.position[1]),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 绘制打击节点 (A) - 橙色星形
        if self.attack_nodes:
            attack_x = [node.position[0] for node in self.attack_nodes]
            attack_y = [node.position[1] for node in self.attack_nodes]
            ax.scatter(attack_x, attack_y, color=node_colors['A'], marker=node_markers['A'],
                      s=250, label=node_labels['A'], edgecolors='black', linewidth=2, zorder=5)
            
            for node in self.attack_nodes:
                ax.annotate(f"{node.node_id}\nAmmo:{node.ammunition}",
                           xy=(node.position[0], node.position[1]),
                           xytext=(0, -25), textcoords='offset points',
                           ha='center', va='top', fontsize=8, fontweight='bold')
                
                # 绘制打击范围（虚线圆）
                """
                attack_range_circle = Circle((node.position[0], node.position[1]),
                                            node.range_km,
                                            color='orange', alpha=0.1, linestyle='--', linewidth=1)
                ax.add_patch(attack_range_circle)
                """
                
        # 绘制评估节点 (E) - 青色倒三角形
        if self.assess_nodes:
            assess_x = [node.position[0] for node in self.assess_nodes]
            assess_y = [node.position[1] for node in self.assess_nodes]
            ax.scatter(assess_x, assess_y, color=node_colors['E'], marker=node_markers['E'],
                      s=200, label=node_labels['E'], edgecolors='black', linewidth=2, zorder=5)
            
            for node in self.assess_nodes:
                ax.annotate(node.node_id,
                           xy=(node.position[0], node.position[1]),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                # 绘制评估范围（虚线圆）
                """
                assessment_circle = Circle((node.position[0], node.position[1]),
                                          node.assessment_radius_km,
                                          color='red', alpha=0.1, linestyle='--', linewidth=1)
                ax.add_patch(assessment_circle)
                """                
        # 添加图例
        #ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=11, framealpha=0.9)
        
        # 为敌方目标添加颜色条
       # if self.targets:
            #cbar = plt.colorbar(scatter_targets, ax=ax, orientation='vertical', pad=0.02)
            #cbar.set_label('目标威胁值', rotation=270, labelpad=20, fontsize=12)
        if self.targets:
            cbar = plt.colorbar(scatter_targets, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label('目标威胁值', rotation=270, labelpad=20, fontsize=12)
        
        # 调整布局
        #plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为右侧图例留出空间
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"初始分布图已保存为: {save_path}")
        
        plt.show()
        return fig
    
    def print_scenario_statistics(self):
        """打印场景统计信息"""
        print("\n" + "="*60)
        print("初始场景统计信息")
        print("="*60)
        
        # 目标统计
        print(f"\n敌方目标统计:")
        print(f"  目标总数: {len(self.targets)}")
        if self.targets:
            avg_threat = np.mean([t.threat_value for t in self.targets])
            max_threat = max([t.threat_value for t in self.targets])
            min_threat = min([t.threat_value for t in self.targets])
            avg_time = np.mean([t.time_to_target_s for t in self.targets])
            print(f"  平均威胁值: {avg_threat:.2f} (范围: {min_threat}-{max_threat})")
            print(f"  平均飞临时间: {avg_time:.1f}秒")
            
            # 打印每个目标的详细信息
            print(f"\n  目标详细信息:")
            for target in self.targets:
                print(f"    {target}")
        
        # 我方节点统计
        print(f"\n我方作战资源统计:")
        total_nodes = len(self.recon_nodes) + len(self.comm_nodes) + len(self.decision_nodes) + \
                      len(self.attack_nodes) + len(self.assess_nodes)
        print(f"  总节点数: {total_nodes}")
        print(f"    侦察节点(O): {len(self.recon_nodes)} 个")
        print(f"    通信节点(C): {len(self.comm_nodes)} 个")
        print(f"    决策节点(D): {len(self.decision_nodes)} 个")
        print(f"    打击节点(A): {len(self.attack_nodes)} 个")
        print(f"    评估节点(E): {len(self.assess_nodes)} 个")
        
        # 打击节点弹药统计
        total_ammunition = sum([node.ammunition for node in self.attack_nodes])
        print(f"\n  打击节点弹药统计:")
        print(f"    总弹药量: {total_ammunition}")
        for node in self.attack_nodes:
            print(f"    {node.node_id}: {node.ammunition} 发弹药，射程: {node.range_km:.1f}km")
        
        # 侦察节点探测能力统计
        if self.recon_nodes:
            avg_detection_radius = np.mean([node.detection_radius_km for node in self.recon_nodes])
            print(f"\n  侦察节点探测能力:")
            print(f"    平均探测半径: {avg_detection_radius:.1f}km")
        # 容量统计
        print(f"\n  节点容量统计:")
        node_groups = [
            ('侦察节点(O)', self.recon_nodes),
            ('通信节点(C)', self.comm_nodes),
            ('决策节点(D)', self.decision_nodes),
            ('打击节点(A)', self.attack_nodes),
            ('评估节点(E)', self.assess_nodes)
        ]      
        for name, nodes in node_groups:
            if nodes:
                capacities = [node.capacity for node in nodes]
                total_capacity = sum(capacities)
                avg_capacity = np.mean(capacities)
                print(f"    {name}: 总容量={total_capacity}, 平均容量={avg_capacity:.1f}")
    
    def build_initial_scenario(self, output_image_path: str = None):
        """构建初始场景：加载数据并绘制分布图"""
        print("="*60)
        print("初始场景构建器")
        print("="*60)
        
        # 1. 读取敌方目标数据
        print("\n1. 读取敌方目标数据...")
        self.load_targets()
        
        # 2. 读取我方节点数据
        print("\n2. 读取我方节点数据...")
        friendly_nodes = self.load_friendly_nodes()
        
        # 3. 绘制初始分布图
        print("\n3. 绘制初始分布图...")
        if output_image_path is None:
            output_image_path = os.path.join(self.data_folder, "initial_scenario_distribution.png")
        
        fig = self.plot_initial_distribution(output_image_path)
        
        # 4. 打印统计信息
        print("\n4. 场景统计信息:")
        self.print_scenario_statistics()
        
        # 5. 返回构建的场景数据
        print("\n5. 场景构建完成！")
        return {
            'targets': self.targets,
            'recon_nodes': self.recon_nodes,
            'comm_nodes': self.comm_nodes,
            'decision_nodes': self.decision_nodes,
            'attack_nodes': self.attack_nodes,
            'assess_nodes': self.assess_nodes,
            'friendly_nodes_dict': friendly_nodes
        }
# 主函数
def main():
    """主函数：构建并展示初始场景"""
    
    # 创建初始场景构建器
    builder = InitialScenarioBuilder(data_folder="primary_situation")  
    # 构建初始场景
    scenario_data = builder.build_initial_scenario()   
    # 示例：访问场景数据
    print("\n" + "="*60)
    print("场景数据访问示例:")
    print("="*60)
    if scenario_data['targets']:
        print(f"第一个目标: {scenario_data['targets'][0]}")
    
    if scenario_data['recon_nodes']:
        print(f"第一个侦察节点: {scenario_data['recon_nodes'][0]}")
    
    if scenario_data['attack_nodes']:
        print(f"第一个打击节点: {scenario_data['attack_nodes'][0]}")
    print("\n场景构建完成，可以开始进行杀伤链构建和优化算法！")   
    return scenario_data
if __name__ == "__main__":
    # 运行主程序
    scenario = main()