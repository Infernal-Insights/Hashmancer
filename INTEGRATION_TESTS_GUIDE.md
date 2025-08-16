# 🧪 End-to-End Integration Tests Guide

Complete guide for running real-world integration tests with hashes.com and Vast.ai integration.

## 🎯 What These Tests Do

The integration tests validate your complete Hashmancer workflow:

1. **📥 Pull Jobs**: Fetch real MD5 jobs from hashes.com
2. **🏗️ Create Jobs**: Convert external jobs to Hashmancer jobs
3. **☁️ Deploy Workers**: Launch cheap Vast.ai GPU workers
4. **🖥️ Local Workers**: Test local worker job assignment
5. **⚡ Process Jobs**: Run jobs for specified duration
6. **🧹 Cleanup**: Automatically clean up all resources

## 🔑 API Key Setup

### Step 1: Get Your API Keys

#### hashes.com API Key
1. Log in to [hashes.com](https://hashes.com)
2. Go to **Account Settings** → **API Keys**
3. Generate a new API key
4. Copy the key (format: `hashes_xxx...`)

#### Vast.ai API Key
1. Log in to [vast.ai](https://vast.ai)
2. Go to **Account** → **API Keys**
3. Generate a new API key
4. Copy the key (format: `xxx...`)

### Step 2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Add the following **Repository secrets**:

```
HASHES_COM_API_KEY=your_hashes_com_api_key_here
VAST_AI_API_KEY=your_vast_ai_api_key_here
```

### Step 3: Configure GitHub Variables (Optional)

Add these **Repository variables** to customize test behavior:

```
ENABLE_E2E_TESTS=true                 # Enable automatic E2E tests
VAST_MAX_COST_PER_HOUR=0.50          # Maximum Vast.ai cost per hour
E2E_TEST_DURATION_MINUTES=15         # Total test duration
E2E_MIN_JOB_RUNTIME_MINUTES=10       # Minimum job runtime required
```

## 🚀 Running Integration Tests

### Option 1: Manual GitHub Actions Run

1. Go to **Actions** → **End-to-End Integration Tests**
2. Click **"Run workflow"**
3. Configure parameters:
   - **Test duration**: How long to run tests (default: 15 minutes)
   - **Min job runtime**: Minimum job processing time (default: 10 minutes)
   - **Vast.ai max cost**: Maximum cost per hour (default: $0.50)
   - **Skip Vast.ai**: Skip cloud worker testing (default: false)

### Option 2: Automatic Triggers

Tests run automatically when:
- **Push to main** with `[e2e]` in commit message
- **Weekly schedule** (Sundays at 2 AM UTC)
- **Integration test files** are modified

### Option 3: Local Testing

```bash
# Copy environment template
cp .env.integration.template .env.integration

# Edit with your API keys
nano .env.integration

# Run the test
./test-integration-e2e.sh
```

## 🎛️ Test Configuration

### Environment Variables

```bash
# Required
HASHES_COM_API_KEY=your_key_here
VAST_AI_API_KEY=your_key_here

# Optional (with defaults)
VAST_MAX_COST_PER_HOUR=0.50          # Max $0.50/hour for Vast.ai
TEST_DURATION_MINUTES=15             # 15 minute total test
MIN_JOB_RUNTIME_MINUTES=10           # 10 minute minimum job runtime
SKIP_VAST_DEPLOYMENT=false           # Set to 'true' to skip Vast.ai testing
```

### Cost Controls

- **Automatic cleanup**: All Vast.ai instances are destroyed after tests
- **Cost limits**: Only instances under `VAST_MAX_COST_PER_HOUR` are used
- **Time limits**: Tests have strict time limits to prevent runaway costs
- **Cheapest instances**: Automatically selects the cheapest available GPU

### Safety Features

- ✅ **Automatic resource cleanup** on test completion or failure
- ✅ **Cost monitoring** and budget limits
- ✅ **Time-based termination** to prevent long-running costs
- ✅ **GPU requirement validation** before deployment
- ✅ **API key validation** before starting expensive operations

## 📊 Test Workflow

### Phase 1: Setup and Validation (2-3 minutes)
- Validate API keys
- Deploy local Hashmancer environment
- Check system prerequisites

### Phase 2: Job Integration (3-5 minutes)
- Pull available MD5 jobs from hashes.com
- Create corresponding Hashmancer jobs
- Verify job queuing in Redis

### Phase 3: Worker Deployment (3-5 minutes)
- Deploy cheapest Vast.ai GPU worker
- Start local test worker
- Verify worker registration

### Phase 4: Job Processing (10-15 minutes)
- Monitor job assignment to workers
- Verify job execution starts
- Ensure minimum runtime requirements
- Monitor worker performance

### Phase 5: Cleanup (1-2 minutes)
- Stop all workers
- Destroy Vast.ai instances
- Clean up test jobs and data
- Generate test report

## 🔍 Monitoring and Debugging

### View Test Progress

1. **GitHub Actions**: Real-time logs in Actions tab
2. **Local logs**: Check `/tmp/hashmancer_integration_test.log`
3. **Worker logs**: Monitor worker output and errors
4. **System monitoring**: GPU usage, memory, network

### Test Artifacts

Automatic collection of:
- **Docker logs**: All container output
- **Worker logs**: Local and remote worker logs
- **Redis diagnostics**: Health and performance data
- **System information**: Hardware and software details
- **GPU information**: NVIDIA GPU status and usage

### Common Issues and Solutions

#### API Key Problems
```bash
# Test hashes.com API
curl -H "Authorization: Bearer $HASHES_COM_API_KEY" https://api.hashes.com/api/v1/account

# Test Vast.ai API  
curl -H "Authorization: Bearer $VAST_AI_API_KEY" https://console.vast.ai/api/v0/users/current/
```

#### No Available Jobs
- Tests create mock jobs if no real jobs available
- Check hashes.com account for available credits

#### Vast.ai Instance Issues
- Instances auto-selected based on cost and availability
- Check Vast.ai account credits and limits
- Use `SKIP_VAST_DEPLOYMENT=true` to test without cloud workers

#### Worker Connection Problems
- Verify server is accessible
- Check firewall settings for worker connections
- Monitor Redis connectivity

## 💰 Cost Estimation

### Typical Costs (15-minute test)
- **Vast.ai GPU Worker**: $0.10 - $0.25 (15 minutes at $0.50/hour max)
- **hashes.com API**: Usually free for basic requests
- **Total estimated cost**: < $0.30 per test run

### Cost Optimization
- **Short test duration**: Default 15 minutes for quick validation
- **Cheapest instances**: Automatically selects lowest-cost GPU
- **Automatic cleanup**: No forgotten instances running up costs
- **Budget limits**: Hard limits prevent overspending

## 🏃 Quick Start Checklist

- [ ] Add `HASHES_COM_API_KEY` to GitHub secrets
- [ ] Add `VAST_AI_API_KEY` to GitHub secrets  
- [ ] Set `ENABLE_E2E_TESTS=true` repository variable (optional)
- [ ] Configure cost limits in repository variables (optional)
- [ ] Run manual test: **Actions** → **End-to-End Integration Tests** → **Run workflow**
- [ ] Monitor test progress and review artifacts
- [ ] Verify automatic cleanup completed

## 🎯 Success Criteria

Your integration test is successful when:

- ✅ **API Authentication**: Both hashes.com and Vast.ai APIs authenticate successfully
- ✅ **Job Integration**: Jobs are pulled from hashes.com and created in Hashmancer
- ✅ **Worker Deployment**: Both local and Vast.ai workers connect to server
- ✅ **Job Assignment**: Jobs are assigned to available workers
- ✅ **Job Execution**: Jobs run for the minimum required duration
- ✅ **Resource Cleanup**: All Vast.ai instances and test data are cleaned up

## 🔗 Triggering Tests

### Commit Message Triggers
```bash
git commit -m "Add new hash algorithm [e2e]"     # Triggers E2E tests
git commit -m "Update worker logic"              # Normal tests only
```

### Manual Triggers
1. GitHub Actions → End-to-End Integration Tests → Run workflow
2. Local: `./test-integration-e2e.sh`

### Scheduled Runs
- **Weekly**: Sundays at 2 AM UTC
- **On changes**: When integration test files are modified

## 📋 Test Report Example

```
📊 End-to-End Integration Test Results
======================================

📋 Test Summary:
  Total tests: 8
  Passed: 8
  Failed: 0
  Success rate: 100.0%

🔄 Test Components:
  ✅ hashes.com API integration
  ✅ Server job creation from external source
  ✅ Vast.ai worker deployment and testing
  ✅ Local worker job assignment
  ✅ End-to-end job processing workflow
  ✅ Worker connectivity validation

🎉 All integration tests passed!

Your Hashmancer system successfully:
  ✅ Pulled jobs from hashes.com
  ✅ Created and queued jobs in the server
  ✅ Deployed and connected Vast.ai workers
  ✅ Assigned jobs to local workers
  ✅ Processed jobs for the required duration
  ✅ Maintained stable worker connections

🚀 Your system is ready for production hash cracking!
```

---

## 🎉 Ready to Test!

Your Hashmancer system now has comprehensive end-to-end testing that validates the complete workflow from job ingestion to processing on both local and cloud workers. The tests are designed to be cost-effective, safe, and provide comprehensive validation of your entire hash cracking infrastructure.

**Happy testing!** 🧪✨