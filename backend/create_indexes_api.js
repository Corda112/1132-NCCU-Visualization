#!/usr/bin/env node
/**
 * 資料庫索引創建腳本
 * 呼叫後端API來創建必要的資料庫索引
 */

const http = require('http');

const callCreateIndexesAPI = () => {
    console.log('🚀 開始呼叫索引創建API...');
    
    const options = {
        hostname: 'localhost',
        port: 3001,
        path: '/api/admin/create-indexes',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    };

    const req = http.request(options, (res) => {
        console.log(`HTTP狀態碼: ${res.statusCode}`);
        
        let data = '';
        
        res.on('data', (chunk) => {
            data += chunk;
        });
        
        res.on('end', () => {
            try {
                const response = JSON.parse(data);
                
                if (res.statusCode === 200) {
                    console.log('✅ 索引創建成功!');
                    console.log('📊 結果:');
                    
                    if (response.results) {
                        response.results.forEach(result => {
                            if (result.status === 'success') {
                                console.log(`  ✅ ${result.index}: ${result.time || '完成'}`);
                            } else {
                                console.log(`  ❌ ${result.index}: ${result.error}`);
                            }
                        });
                    }
                    
                    if (response.indexes) {
                        console.log('\n📋 創建的索引:');
                        response.indexes.forEach(index => {
                            console.log(`  - ${index.name}`);
                        });
                    }
                    
                    console.log('\n🎉 資料庫效能優化完成!');
                    console.log('建議: 重新啟動前端服務以獲得最佳效能');
                    
                } else {
                    console.error('❌ API呼叫失敗:');
                    console.error(response);
                }
                
            } catch (error) {
                console.error('❌ 回應解析失敗:', error.message);
                console.error('原始回應:', data);
            }
        });
    });

    req.on('error', (error) => {
        if (error.code === 'ECONNREFUSED') {
            console.error('❌ 無法連接到後端服務器');
            console.error('請確認後端服務正在 http://localhost:3001 運行');
            console.error('可以使用以下命令啟動後端:');
            console.error('  cd backend && npm start');
        } else {
            console.error('❌ 請求錯誤:', error.message);
        }
    });

    req.on('timeout', () => {
        console.error('❌ 請求超時');
        req.destroy();
    });

    // 設置超時
    req.setTimeout(30000); // 30秒超時
    
    // 結束請求
    req.end();
};

// 檢查伺服器是否運行
const checkServer = () => {
    console.log('🔍 檢查後端服務狀態...');
    
    const options = {
        hostname: 'localhost',
        port: 3001,
        path: '/health',
        method: 'GET',
        timeout: 5000
    };

    const req = http.request(options, (res) => {
        if (res.statusCode === 200) {
            console.log('✅ 後端服務運行正常');
            // 延遲1秒再呼叫索引API
            setTimeout(callCreateIndexesAPI, 1000);
        } else {
            console.error(`❌ 後端服務回應異常: ${res.statusCode}`);
        }
    });

    req.on('error', (error) => {
        console.error('❌ 後端服務未運行或無法連接');
        console.error('請先啟動後端服務:');
        console.error('  cd backend && npm start');
        process.exit(1);
    });

    req.setTimeout(5000, () => {
        console.error('❌ 健康檢查超時');
        req.destroy();
        process.exit(1);
    });

    req.end();
};

if (require.main === module) {
    console.log('🛠️  資料庫索引優化工具');
    console.log('=' * 40);
    
    checkServer();
} 