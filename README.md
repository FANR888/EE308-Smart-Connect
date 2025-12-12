# EE308-Smart-Connect
An intelligent address book that allows user management and uses AI to automatically extract information and assist in adding new contacts.


# 🚀 快速开始
按照以下步骤在本地运行项目：

## 1. 克隆项目
Bash

git clone https://github.com/your-username/gemini-smart-connect.git
cd gemini-smart-connect’‘’
## 2. 安装依赖
'''Bash
npm install'''
## 3. 配置环境变量
在项目根目录下创建一个 .env (或者 .env.local) 文件，并添加您的 Google Gemini API Key：

Code snippet

GEMINI_API_KEY=your_api_key_here
注意: 您可以从 Google AI Studio 获取 API Key。

## 4. 运行开发服务器
Bash

npm run dev
打开浏览器访问 http://localhost:3000 (或终端显示的端口)。

# 📖 使用指南
登录与注册
用户模式: 默认选择 "User" 标签。点击底部的 "Need an account? Create one" 进行注册。

管理员模式: 点击 "Admin" 标签登录（需预先配置管理员账号或在代码中硬编码初始管理员）。

联系人管理
点击右上角的 Export/Import 图标进行 Excel 数据的批量操作。

点击 Add Contact 按钮打开表单，填写详细信息。

点击联系人卡片上的 星星图标 将其加入收藏夹。

# 📂 项目结构
Plaintext

src/
├── components/      # UI 组件 (ContactForm 等)
├── services/        # 业务逻辑 (StorageService, ExcelService, AIService)
├── types/           # TypeScript 类型定义
├── App.tsx          # 主应用逻辑与路由
├── index.tsx        # 入口文件
└── index.css        # Tailwind 样式引入
