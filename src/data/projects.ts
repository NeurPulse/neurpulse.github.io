// 在这里维护你的项目列表，页面会自动渲染
export interface Project {
  title: string;
  description: string;
  tech: string[];
  github?: string;
  year?: string;
}

export const projects: Project[] = [
  {
    title: '基于 LSTM 的股票价格预测',
    description:
      '使用 LSTM 网络对股票收盘价进行时间序列预测，包含滑动窗口构造、归一化与反归一化、与 ARIMA 基线的对比实验。',
    tech: ['Python', 'PyTorch', 'Pandas'],
    github: 'https://github.com/your-username/your-repo',
    year: '2026',
  },
  {
    title: '蒙特卡洛期权定价工具',
    description:
      '实现了几何布朗运动路径模拟、欧式/亚式期权定价，以及方差缩减技术（对偶变量、控制变量）的效果对比。',
    tech: ['Python', 'NumPy', 'Matplotlib'],
    github: 'https://github.com/your-username/your-repo',
    year: '2026',
  },
  {
    title: 'Black-Scholes 偏微分方程数值解',
    description:
      '用有限差分法（显式/隐式/Crank-Nicolson）求解 BS 方程，与解析解对比验证收敛阶。',
    tech: ['Python', 'NumPy', 'SciPy'],
    github: 'https://github.com/your-username/your-repo',
    year: '2025',
  },
];
