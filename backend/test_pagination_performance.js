const http = require('http');

function makeRequest(path) {
    return new Promise((resolve, reject) => {
        const options = {
            hostname: 'localhost',
            port: 3001,
            path: path,
            method: 'GET'
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => {
                data += chunk;
            });
            res.on('end', () => {
                if (res.statusCode >= 200 && res.statusCode < 300) {
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        reject(new Error('Failed to parse JSON response.'));
                    }
                } else {
                    reject(new Error(`Request failed with status code ${res.statusCode}. Response: ${data}`));
                }
            });
        });

        req.on('error', (e) => {
            reject(new Error(`Request error: ${e.message}`));
        });

        req.end();
    });
}

async function testPaginationPerformance() {
    console.log('🚀 開始測試分頁API效能...');
    
    // 檢查服務器是否可用
    console.log('🔍 檢查服務器狀態...');
    try {
        await makeRequest('/health');
        console.log('✅ 服務器連接正常');
    } catch (err) {
        console.error('❌ 服務器連接失敗:', err.message);
        console.log('💡 請確認後端服務器已啟動: 在 backend 目錄下執行 node server.js');
        return;
    }

    // 測試1: 首頁查詢 (會執行COUNT)
    console.log('\\n📊 測試1: 首頁查詢 (包含COUNT)');
    const start1 = Date.now();
    try {
        const data1 = await makeRequest('/api/articles?page=1&limit=10');
        const time1 = Date.now() - start1;
        
        console.log(`✅ 首頁查詢: ${time1}ms`);
        console.log(`   資料筆數: ${data1.articles?.length || 0}`);
        console.log(`   總頁數: ${data1.pagination?.totalPages || data1.totalPages || 'N/A'}`);
        console.log(`   有更多: ${data1.pagination?.hasMore || 'N/A'}`);
        if (data1.performance) {
            console.log(`   查詢時間: ${data1.performance.queryTime}ms`);
            console.log(`   COUNT時間: ${data1.performance.countTime || 'N/A'}ms`);
        }
    } catch (err) {
        console.error('❌ 首頁查詢失敗:', err.message);
    }
    
    // 等待一下，避免請求過快
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // 測試2: 第二頁查詢 (不執行COUNT)
    console.log('\\n📄 測試2: 第二頁查詢 (無COUNT)');
    const start2 = Date.now();
    try {
        const data2 = await makeRequest('/api/articles?page=2&limit=10');
        const time2 = Date.now() - start2;
        
        console.log(`✅ 第二頁查詢: ${time2}ms`);
        console.log(`   資料筆數: ${data2.articles?.length || 0}`);
        console.log(`   總數: ${data2.pagination?.totalCount !== undefined ? data2.pagination.totalCount : '未計算'}`);
        if (data2.performance) {
            console.log(`   查詢時間: ${data2.performance.queryTime}ms`);
            console.log(`   COUNT時間: ${data2.performance.countTime !== undefined ? data2.performance.countTime + 'ms' : '未執行'}`);
        }
    } catch (err) {
        console.error('❌ 第二頁查詢失敗:', err.message);
    }

    // 測試3: 緩存測試 (重複首頁查詢)
    console.log('\\n💾 測試3: 緩存測試 (重複首頁查詢)');
    const start3 = Date.now();
    try {
        const data3 = await makeRequest('/api/articles?page=1&limit=10');
        const time3 = Date.now() - start3;
        
        console.log(`✅ 緩存查詢: ${time3}ms`);
        console.log(`   資料筆數: ${data3.articles?.length || 0}`);
        if (data3.performance) {
            console.log(`   查詢時間: ${data3.performance.queryTime}ms (應與第一次查詢相同)`);
        }
    } catch (err) {
        console.error('❌ 緩存查詢失敗:', err.message);
    }
    
    console.log('\\n🎯 測試完成！');
}

if (require.main === module) {
    testPaginationPerformance().catch(err => {
        console.error("測試腳本發生嚴重錯誤:", err);
    });
}

module.exports = testPaginationPerformance; 