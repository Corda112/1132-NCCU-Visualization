#!/usr/bin/env node
/**
 * 驗證所有修正是否正確的測試腳本
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 驗證資料庫優化修正...');
console.log('='.repeat(50));

const checkResults = [];

// 1. 檢查路徑常數定義
function checkPathConstants() {
    console.log('\n📂 檢查路徑常數定義...');
    
    const files = [
        'add_sqlite.py',
        'optimize_database.py'
    ];
    
    for (const file of files) {
        const filePath = path.join(__dirname, file);
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf8');
            
            const hasScriptDir = content.includes('SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))');
            const hasDbPath = content.includes('DB_PATH = os.path.join(SCRIPT_DIR, \'db.sqlite3\')');
            const usesDbPath = content.includes('db_path = DB_PATH');
            
            if (hasScriptDir && hasDbPath && usesDbPath) {
                console.log(`  ✅ ${file}: 路徑常數正確定義`);
                checkResults.push({ file, test: '路徑常數', status: 'pass' });
            } else {
                console.log(`  ❌ ${file}: 路徑常數定義有問題`);
                checkResults.push({ file, test: '路徑常數', status: 'fail' });
            }
        } else {
            console.log(`  ⚠️  ${file}: 檔案不存在`);
            checkResults.push({ file, test: '路徑常數', status: 'missing' });
        }
    }
}

// 2. 檢查JS字串乘法修正
function checkStringRepeat() {
    console.log('\n🔤 檢查JavaScript字串repeat修正...');
    
    const files = [
        'create_indexes_api.js',
        'performance_test.js'
    ];
    
    for (const file of files) {
        const filePath = path.join(__dirname, file);
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf8');
            
            const hasIncorrectSyntax = content.includes("'=' * ");
            const hasCorrectSyntax = content.includes('.repeat(');
            
            if (!hasIncorrectSyntax && hasCorrectSyntax) {
                console.log(`  ✅ ${file}: 字串repeat語法正確`);
                checkResults.push({ file, test: '字串repeat', status: 'pass' });
            } else if (hasIncorrectSyntax) {
                console.log(`  ❌ ${file}: 仍有錯誤的字串乘法語法`);
                checkResults.push({ file, test: '字串repeat', status: 'fail' });
            } else {
                console.log(`  ⚠️  ${file}: 找不到字串repeat語法`);
                checkResults.push({ file, test: '字串repeat', status: 'unknown' });
            }
        } else {
            console.log(`  ⚠️  ${file}: 檔案不存在`);
            checkResults.push({ file, test: '字串repeat', status: 'missing' });
        }
    }
}

// 3. 檢查索引定義一致性
function checkIndexConsistency() {
    console.log('\n🔗 檢查索引定義一致性...');
    
    const serverPath = path.join(__dirname, 'server.js');
    const optimizePath = path.join(__dirname, 'optimize_database.py');
    
    if (fs.existsSync(serverPath) && fs.existsSync(optimizePath)) {
        const serverContent = fs.readFileSync(serverPath, 'utf8');
        const optimizeContent = fs.readFileSync(optimizePath, 'utf8');
        
        // 提取索引名稱
        const serverIndexes = (serverContent.match(/idx_\w+/g) || []).filter((v, i, a) => a.indexOf(v) === i);
        const optimizeIndexes = (optimizeContent.match(/idx_\w+/g) || []).filter((v, i, a) => a.indexOf(v) === i);
        
        console.log(`  Server.js 索引: ${serverIndexes.join(', ')}`);
        console.log(`  optimize_database.py 索引: ${optimizeIndexes.join(', ')}`);
        
        const isConsistent = JSON.stringify(serverIndexes.sort()) === JSON.stringify(optimizeIndexes.sort());
        
        if (isConsistent) {
            console.log(`  ✅ 索引定義一致 (${serverIndexes.length}個索引)`);
            checkResults.push({ file: 'server.js + optimize_database.py', test: '索引一致性', status: 'pass' });
        } else {
            console.log(`  ❌ 索引定義不一致`);
            checkResults.push({ file: 'server.js + optimize_database.py', test: '索引一致性', status: 'fail' });
        }
    } else {
        console.log(`  ⚠️  缺少必要檔案`);
        checkResults.push({ file: 'server.js + optimize_database.py', test: '索引一致性', status: 'missing' });
    }
}

// 4. 檢查async/await修正
function checkAsyncAwaitFix() {
    console.log('\n⚡ 檢查async/await修正...');
    
    const serverPath = path.join(__dirname, 'server.js');
    if (fs.existsSync(serverPath)) {
        const content = fs.readFileSync(serverPath, 'utf8');
        
        // 檢查是否移除了async callback
        const hasAsyncCallback = content.includes('async (err) => {');
        const hasPromiseChain = content.includes('createPerformanceIndexes()\n                    .then(');
        
        if (!hasAsyncCallback && hasPromiseChain) {
            console.log(`  ✅ server.js: async/await修正正確`);
            checkResults.push({ file: 'server.js', test: 'async/await修正', status: 'pass' });
        } else {
            console.log(`  ❌ server.js: async/await修正有問題`);
            console.log(`    - 移除async callback: ${!hasAsyncCallback}`);
            console.log(`    - 使用Promise chain: ${hasPromiseChain}`);
            checkResults.push({ file: 'server.js', test: 'async/await修正', status: 'fail' });
        }
    } else {
        console.log(`  ⚠️  server.js檔案不存在`);
        checkResults.push({ file: 'server.js', test: 'async/await修正', status: 'missing' });
    }
}

// 5. 檢查文件完整性
function checkFileCompleteness() {
    console.log('\n📄 檢查文件完整性...');
    
    const requiredFiles = [
        'optimize_database.py',
        'create_indexes_api.js',
        'performance_test.js',
        'create_indexes.sql',
        'DATABASE_OPTIMIZATION.md',
        'add_sqlite.py',
        'server.js'
    ];
    
    for (const file of requiredFiles) {
        const filePath = path.join(__dirname, file);
        if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            console.log(`  ✅ ${file}: 存在 (${stats.size} bytes)`);
            checkResults.push({ file, test: '檔案存在', status: 'pass' });
        } else {
            console.log(`  ❌ ${file}: 不存在`);
            checkResults.push({ file, test: '檔案存在', status: 'fail' });
        }
    }
}

// 生成總結報告
function generateReport() {
    console.log('\n📊 驗證結果總結');
    console.log('='.repeat(50));
    
    const totalTests = checkResults.length;
    const passedTests = checkResults.filter(r => r.status === 'pass').length;
    const failedTests = checkResults.filter(r => r.status === 'fail').length;
    const missingTests = checkResults.filter(r => r.status === 'missing').length;
    
    console.log(`總測試項目: ${totalTests}`);
    console.log(`✅ 通過: ${passedTests}`);
    console.log(`❌ 失敗: ${failedTests}`);
    console.log(`⚠️  缺失: ${missingTests}`);
    
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);
    console.log(`\n成功率: ${successRate}%`);
    
    if (failedTests > 0) {
        console.log('\n❌ 失敗的測試:');
        checkResults
            .filter(r => r.status === 'fail')
            .forEach(r => console.log(`  - ${r.file}: ${r.test}`));
    }
    
    if (missingTests > 0) {
        console.log('\n⚠️  缺失的檔案:');
        checkResults
            .filter(r => r.status === 'missing')
            .forEach(r => console.log(`  - ${r.file}: ${r.test}`));
    }
    
    if (failedTests === 0 && missingTests === 0) {
        console.log('\n🎉 所有修正都已正確實施！');
        console.log('\n📌 建議下一步:');
        console.log('  1. 重啟後端服務測試索引自動創建');
        console.log('  2. 執行效能測試驗證優化效果');
        console.log('  3. 測試前端圖表點擊功能');
    } else {
        console.log('\n⚠️  還有待修正的問題，請檢查上述失敗項目');
    }
}

// 執行所有檢查
checkPathConstants();
checkStringRepeat();
checkIndexConsistency();
checkAsyncAwaitFix();
checkFileCompleteness();
generateReport(); 