little skill

1、判断 BST 的合法性：对于某一个节点`root`，他只能管得了自己的左右子节点，怎么把`root`的约束传递给左右子树呢？通过使用辅助函数，增加函数参数列表，在参数中携带额外信息，将这种约束传递给子树的所有节点

### 1、二分法模板

版本1
当我们将区间[l, r]划分成[l, mid]和[mid + 1, r]时，其更新操作是r = mid或者l = mid + 1;，计算mid时不需要加1。

```c++
//答案在左边
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```

版本2
当我们将区间[l, r]划分成[l, mid - 1]和[mid, r]时，其更新操作是r = mid - 1或者l = mid;，此时为了防止死循环，计算mid时需要加1。

```c++
//答案在右边这一侧
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

### 2、shared_ptr智能指针实现

```c++
template<typename T>
class SharedPtr
{
public:
	 SharedPtr(T* ptr = NULL):_ptr(ptr), _pcount(new int(1))
	 {}
	 SharedPtr(const SharedPtr& s):_ptr(s._ptr), _pcount(s._pcount)  //拷贝构造
	 {
		*(_pcount)++;
	 }
	 SharedPtr<T>& operator=(const SharedPtr& s)  //拷贝赋值
	 {
		 if (this != &s)
		 {
			if (--(*(this->_pcount)) == 0)
			{
				 delete this->_ptr;
				 delete this->_pcount;
			}
			 _ptr = s._ptr;
			 _pcount = s._pcount;
			 *(_pcount)++;
		 }
		return *this;
	 }
	 T& operator*()
	 {
		return *(this->_ptr);
	 }
	 T* operator->()
	 {
		return this->_ptr;
	 }
	 ~SharedPtr()
	 {
		 --(*(this->_pcount));
		 if (this->_pcount == 0)
		 {
			 delete _ptr;
			 _ptr = NULL;
			 delete _pcount;
			 _pcount = NULL;
		 }
	 }
private:
	 T* _ptr;
	 int* _pcount;//指向引用计数的指针
};
```

### 3、归并排序

```c++
void merge(vector<int>& nums,vector<int>& tmp,int l,int r)
    {
        if(l>=r) return;
        int mid=l+(r-l)/2;
        merge(nums,tmp,l,mid);
        merge(nums,tmp,mid+1,r);
        int pos1=l;
        int pos2=mid+1;
        int p=l;

        while(pos1<=mid&&pos2<=r)
        {
            if(nums[pos1]<=nums[pos2]) tmp[p++]=nums[pos1++];
            else
            {
                tmp[p++]=nums[pos2++];
            }
        }
        while(pos1<=mid) tmp[p++]=nums[pos1++];
        while(pos2<=r) tmp[p++]=nums[pos2++];
        for(int i=l;i<=r;i++) nums[i]=tmp[i];
    }
```

### 4、快排排序

```c++
/*
选取一个基准元素（pivot）
比pivot小的放到pivot左边，比pivot大的放到pivot右边
对pivot左边的序列和右边的序列分别递归的执行步骤1和步骤2
*/
//快速排序采用的思想是分治思想
//时间复杂度为O(N*logN)
void quicksort(vector<int>& nums,int startindex,int endindex)
{
	if(startindex>=endindex) return;
	int privo=partition(nums,startindex,endindex);
	quicksort(nums,startindex,privo-1);
	quicksort(nums,privo+1,endindex);
}
int partition(vector<int>& nums,int startindex,int endindex)
{
	int pri=nums[startindex];
	int left=startindex;
	int right=endindex;
	while(left!=right)
	{
		while(left<right&&nums[right]>pri)
			right--;
		while(left<right&&nums[left]<=pri)
			left++;
		if(left<right)
		{
			int p=nums[left];
			nums[left]=nums[right];
			nums[right]=p;
		}
	}
    if(left == right && nums[right] > pri) right--; //尽可能的缩小
	nums[startindex]=nums[left];   //把基准值移到分界线出
	nums[left]=pri;  //因为刚开始时选的第一个
	return left;   //返回基准值的下标
}

```



## 数学

### 1、两数之和（哈希表）

```c++
//暴力解法
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> v;
        for(int i=0;i<nums.size();i++)
        {
            for(int j=i+1;j<nums.size();j++)
            {
                if(nums[i]+nums[j]==target)
                {
                    v.push_back(i);
                    v.push_back(j);

                }
            }
        }
        return v;

    }
};
//哈希表
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable;
        for (int i = 0; i < nums.size(); ++i) {
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end()) {
                return {it->second, i};  //it->second：返回键值对的第二个值
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};

```

### 167、两数之和||---输入有序数组

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int j=numbers.size()-1;
        int i=0;
        vector<int> rs;
        while(i<j)
        {
            if(numbers[i]+numbers[j]<target)
            {
                i++;
            }
            else if(numbers[i]+numbers[j]>target)
            {
                j--;
            }
            else
            {
                return vector<int>{i+1,j+1};
            }
        }
        return rs;
    }
};
```



### 15、三数之和（双指针+排序）

```c++
//双指针+排序
//思路：三指针，首先i从下标0开始指，left\right在i之后的两端，注意一点是：消除重复元素
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        
        sort(nums.begin(),nums.end());  //排序的方法可以跳过重复的元素
        for(int i=0;i<nums.size();i++)
        {
            if(nums[0]>0)
            {
                break;
            }
            if(i>0&&nums[i]==nums[i-1])  //去除i的可重复元素
            {
                continue;
            }
            int left=i+1;
            int right=nums.size()-1;
            while(right>left)
            {
                if(nums[i]+nums[left]+nums[right]>0)
                {
                    right--;
                }
                else if(nums[i]+nums[left]+nums[right]<0)
                {
                    left++;
                }
                else
                {
                    result.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    while((left<right)&&(nums[left]==nums[left+1])) left++;  
                    while((left<right)&&(nums[right]==nums[right-1])) right--;
                    left++;
                    right--;
                }
            }
            
        }
        return result;
    }
};
```

### 405、数字转成16进制

```c++
class Solution {
public:
    string toHex(int num) {
       if (num == 0)
        {
            return "0";
        }
        // 0-15映射成字符
        string num2char = "0123456789abcdef";
        // 转为为非符号数
        unsigned int n = num;    //因为c++里不支持对负数做移位 所以要转换为非负数
        string res = "";
        while (n > 0)
        {
            res = num2char[n&15] + res;
            n >>= 4;
        }
        return res;
    }
};
```



### 16、最接近的三数之和

```c++
//暴力
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int ans=nums[0]+nums[1]+nums[2];
        int n=nums.size();
        for(int i=0;i<n;i++)
        {
            for(int j=i+1;j<n;j++)
            {
                for(int k=j+1;k<n;k++)
                {
                    if(abs(nums[i]+nums[j]+nums[k]-target)<abs(ans-target))
                        ans=nums[i]+nums[j]+nums[k];
                }
            }
        }
        return ans;
    }
};
/*
首先将 nums 数组排序，然后固定一重循环枚举起始位置 i。
然后初始 l = i + 1，r = n - 1；如果发现 nums[i] + nums[l] + nums[r] == target，则可以直接返回 target；
若发现 nums[i] + nums[l] + nums[r] < target，则 l++；否则 r--；
直到 l>=r 停止，继续向后增加初始位置 i。
*/
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int ans=nums[0]+nums[1]+nums[2];
        int n=nums.size();
        sort(nums.begin(), nums.end());
        for(int i=0;i<n;i++)
        {
            int left=i+1;
            int right=n-1;
            while(left<right)
            {
                if(abs(nums[i]+nums[left]+nums[right]-target)<abs(ans-target))
                {
                    ans=nums[i]+nums[left]+nums[right];
                }
                if(nums[i]+nums[left]+nums[right]==target)
                    return target;
                else if (nums[i]+nums[left]+nums[right] < target)
                    left++;
                else
                    right--;
            }
        }
        return ans;
    }
};

```



### 18、四数相加

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        sort(nums.begin(),nums.end());

        for(int k=0;k<nums.size();k++)
        {
            if(k>0&&(nums[k]==nums[k-1]))
            {
                continue;
            }
            for(int i=k+1;i<nums.size();i++)
            {
                if(i>k+1&&(nums[i]==nums[i-1]))
                {
                    continue;
                }
                int left=i+1;
                int right=nums.size()-1;
                while(right>left)
                {
                    if (nums[k] + nums[i] + nums[left] + nums[right] > target) 
                    {
                        right--;
                    } 
                    else if (nums[k] + nums[i] + nums[left] + nums[right] < target) 
                    {
                        left++;
                    } 
                    else 
                    {
                        result.push_back(vector<int>{nums[k], nums[i], nums[left], nums[right]});
                        // 去重逻辑应该放在找到一个四元组之后
                        while (right > left && nums[right] == nums[right - 1]) right--;
                        while (right > left && nums[left] == nums[left + 1]) left++;

                        // 找到答案时，双指针同时收缩
                        right--;
                        left++;
                    }
                }

             }
            
        }return result;
    }
    
};
//回溯法
class Solution {
private:
    vector<vector<int>> ans;
    vector<int> myNums, subans;
    int tar, numSize;
    void DFS(int low, int sum) {
        if (sum == tar && subans.size() == 4) {
            ans.emplace_back(subans);
            return;
        }
        for (int i = low; i < numSize; ++i) {
            if (numSize - i < int(4 - subans.size())) { //剪枝
                return;
            }
            if (i > low && myNums[i] == myNums[i - 1]) { //去重
                continue; 
            }
            if (i < numSize - 1 && sum + myNums[i] + int(3 - subans.size()) * myNums[i + 1] > tar) { //剪枝
                return;
            }
            if (i < numSize - 1 && sum + myNums[i] + int(3 - subans.size()) * myNums[numSize - 1] < tar) { //剪枝
                continue;
            }
            subans.emplace_back(myNums[i]);
            DFS(i + 1, myNums[i] + sum);
            subans.pop_back();
        }
        return;
    }
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        myNums = nums;
        tar = target;
        numSize = nums.size();
        if (numSize < 4) {
            return ans;
        }
        DFS(0, 0);
        return ans;    
    }
};
 
```

### 224、基本计算器

```c++
/*
如果当前是数字，那么更新计算当前数字；
如果当前是操作符+或者-，那么需要更新计算当前计算的结果 res，并把当前数字 num 设为 0，sign 设为正负，重新开始；
如果当前是 ( ，那么说明遇到了右边的表达式，而后面的小括号里的内容需要优先计算，所以要把 res，sign 进栈，更新 res 和 sign 为新的开始；
如果当前是 ) ，那么说明右边的表达式结束，即当前括号里的内容已经计算完毕，所以要把之前的结果出栈，然后计算整个式子的结果；
最后，当所有数字结束的时候，需要把最后的一个 num 也更新到 res 中。

*/
class Solution {
public:
    int calculate(string s) {
        stack<int> st;
        int num=0,res=0,sign=1;
        for(auto& c:s)
        {
            if(isdigit(c))
            {
                num=num*10+(c-'0');
            }
            else if(c=='+'||c=='-')
            {
                res+=num*sign;
                num=0;
                sign=(c=='+')?1:-1;
            }
            else if(c=='(')
            {
                st.push(res);
                st.push(sign);
                res=0;
                sign=1;
            }
            else if(c==')')
            {
                res+=num*sign;
                num=0;
                res*=st.top(); st.pop();
                res+=st.top(); st.pop();
            }
        }
        res+=num*sign; //比如"1+1"这种情况
        return res;
    }
};
```



### 227、基本计算器||

```c++
//基本思路
/*
对于加减号后的数字，将其直接压入栈中
对于乘除号后的数字，可以直接与栈顶元素计算，并替换栈顶元素为计算后的结果
并用变量 preSign 记录每个数字之前的运算符，对于第一个数字，其之前的运算符视为加号
*/
class Solution {
public:
    int calculate(string s) {
        vector<int> stk;
        char preSign = '+';
        int num = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) 
		{
            if (isdigit(s[i])) 
			{
                num = num * 10 + int(s[i] - '0');
            }
            if (!isdigit(s[i]) && s[i] != ' ' || i == n - 1) 
			{
                switch (preSign) 
				{
                    case '+':
                        stk.push_back(num);
                        break;
                    case '-':
                        stk.push_back(-num);
                        break;
                    case '*':
                        stk.back() *= num;
                        break;
                    default:
                        stk.back() /= num;
                }
                preSign = s[i];
                num = 0;
            }
        }
        return accumulate(stk.begin(), stk.end(), 0);
    }
};
```

### 400、第N位数字

```c++
class Solution {
public:
    int findNthDigit(int n) {
        //求n所落在数值的位数
        long digit=1,start=1,count=9;
        while(n>count)
        {
            n-=count;
            start*=10;	//这个位数开始的数字
            digit+=1;	//代表几位数
            count=9*start*digit;	//这个位数共有多少个数
        }
        long num=start+(n-1)/digit;	//找到第几个数字
        int res=to_string(num)[(n-1)%digit]-'0';	//转换为数组，再找确切的第几位
        return res;

    }
};
```



### 9、回文数

解题思路：把数从末尾开始取，依此存到容器中，利用容器的size()函数进行比较

还有一种方法和【7】一样，但是当初没有考虑到溢出的问题

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        if(x<0) return false;
        vector<int> res;
        while(x)
        {
            int n=x%10;
            res.push_back(n);
            x=x/10;
        }
        int m=res.size();
        for(int i=0;i<m;i++)
        {
            if(res[i]!=res[m-i-1])
                return false;
        }
        return true;
    }
};
//不需要开辟空间的做法
//如果反转的话会出现溢出的问题
//思路：我们只需计算出后一半的逆序值，再判断是否和前一半相等
class Solution {
public:
    bool isPalindrome(int x) {
        if(x<0||(x&&x%10==0)) return false;   //防止出现11110这种情况
        int s=0;
        while(s<=x)
        {
            s=s*10+x%10;
            if(s==x||s==x/10) return true;
            x=x/10;
        }
        return false;
    }
};
```

### 7、整数反转

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 `[−2^31, 2^31 − 1]`；也就是说我们不能用`long`存储最终结果，而且有些数字可能是合法范围内的数字，但是**反转**过来就超过范围了。

上图中，绿色的是最大32位整数
第二排数字中，橘子的是5，它是大于上面同位置的4，这就意味着5后跟任何数字，都会比最大32为整数都大。
所以，我们到【最大数的1/10】时，就要开始判断了
如果某个数字大于 214748364那后面就不用再判断了，肯定溢出了。
如果某个数字等于 214748364呢，这对应到上图中第三、第四、第五排的数字，需要要跟最大数的末尾数字比较，如果这个数字比7还大，说明溢出了。

对于负数也是一样：

![image-20201020212640645](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20201020212640645.png)

上图中绿色部分是最小的32位整数，同样是在【最小数的 1/10】时开始判断
如果某个数字小于 -214748364说明溢出了
如果某个数字等于 -214748364，还需要跟最小数的末尾比较，即看它是否小于8

```c++
class Solution {
public:
    int reverse(int x) {   
    int n;
	long long y=0;
	while (x)
	{
		n = x % 10;
		//防止数据的溢出
        if (y > INT_MAX/10 || (y == INT_MAX / 10 && n > 7)) return 0;
        if (y < INT_MIN/10 || (y == INT_MIN / 10 && n < -8)) return 0;
		y=(y*10)+n;
		x = x / 10;
    }
    return y;
    }
};
```

### 31、下一个排列

```c++
1、从数组末尾往前找，找到第一个位置j，使得nums[j]<nums[j+1]
2、如果不存在这样的j,则说明数组不是递增的，将数组逆转即可
3、如果存在这样的j，则从末尾找到第一个位置i>j，使得nums[i]>nums[j]
4、交换nums[i]和nums[j],然后将数组从j+1到末尾部分逆转
    
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n=nums.size();
        int j=-1;
        for(int i=n-2;i>=0;i--)
        {
            if(nums[i]<nums[i+1])
            {
                j=i;
                break;
            }
        }
        if(j==-1) 
            reverse(nums.begin(),nums.end());
        else
        {
            for(int i=n-1;i>j;i--)
            {
                if(nums[i]>nums[j])
                {
                    swap(nums[i],nums[j]);
                    break;
                }
            }
            reverse(nums.begin()+j+1,nums.end());
        }
    }
};
```

### 556、下一个更大元素|||

```c++
//思路和上边[31、下一个排列]相同
class Solution {
public:
    int nextGreaterElement(int n) {
        string s;
        while(n)
        {
            s+=(char)(n%10+'0');
            n/=10;
        }
        reverse(s.begin(),s.end());

        int l=s.length();
        int j=-1;
        for(int i=l-2;i>=0;i--)
        {
            if(s[i]<s[i+1])
            {
                j=i;
                break;
            }
        }
        if(j==-1) 
            return -1;
        else
        {
            for(int i=l-1;i>j;i--)
            {
                if(s[i]>s[j])
                {
                    swap(s[i],s[j]);
                    break;
                }
            }
            reverse(s.begin()+j+1,s.end());
        }
        long long res=0;
        for(int i=0;i<l;i++)
        {
            res=res*10+s[i]-'0';

        }
        if(res>=1ll<<31)
            res=-1;
        return res;
    }
}
```

### 4、寻找两个正序数组的中位数(递归)

```c++
/*
我们将问题直接转化成求两个数组中第k小的元素，和题解1类似，只不过这次我们从另一种角度去考虑。

首先我们给出getKthSmallest函数的参数含义：表示从nums1[i:nums1.size() - 1]和nums2[j:nums2.size() - 1]这两个切片中找到第k小的数字。假设nums1和nums2的切片元素个数分别为len1和len2，为了方便讨论，我们定义为len1 < len2。

考虑一般的情况，我们在这两个切片中各取k / 2个元素，令si = i + k / 2, sj = j + 2 / k，得到切片nums1[i : si - 1]和nums2[j : sj - 1]。

如果nums1[si - 1] < nums2[sj - 1]说明nums1[i : si - 1]中的元素都是小于第k小的元素的，我们可以舍去这部分元素，在剩下的区间内去找第k - k / 2小的元素。

如果nums1[si - 1] >= nums2[sj - 1]说明nums2[j : sj - 1]中的元素都是小于第k小的元素的，我们可以舍去这部分元素，在剩下的区间内去找第k - k / 2小的元素。

考虑特殊情况，当nums1[i : nums1.size() - 1]切片中的元素小于k / 2个了，那么我们就将这个切片中的所有元素取出来，在nums2中仍然取k / 2个元素，这是因为k是一个有效值即len1 + len2 >= k，不可能出现len1 < k / 2的同时len2 < k / 2；另一方面因为我们确保了len1 < len2，所以也不可能出现len1 >= k / 2的时候，len2 < k / 2，即len2 >= k / 2恒成立。
*/
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int tot=nums1.size()+nums2.size();
        if(tot%2==0)
        {
            int left=find(nums1,0,nums2,0,tot/2);
            int right=find(nums1,0,nums2,0,tot/2+1);
            return (left+right)/2.0;
        }
        else
        {
            return find(nums1,0,nums2,0,tot/2+1);
        }
    }
    int find(vector<int>& nums1,int i,vector<int>& nums2,int j,int k)
    {
        if(nums1.size()-i>nums2.size()-j) return find(nums2,j,nums1,i,k);
        if(i==nums1.size()) return nums2[j+k-1];
        if(k==1) 
        {
            if(i==nums1.size()) return nums2[j];
            else return min(nums1[i],nums2[j]);
        }
        int si=min(i+k/2,(int)nums1.size());
        int sj=j+k/2;
        if(nums1[si-1]>nums2[sj-1])
        {
            return find(nums1,i,nums2,sj,k-(sj-j));
        }
        else{
            return find(nums1,si,nums2,j,k-(si-i));
        }

    }
};
```



### 1480、一维数组的动态和

```
class Solution {
public:
    vector<int> runningSum(vector<int>& nums) 
    {
        vector<int> v;
        v.push_back(nums[0]);
        for(int i=1;i<nums.size();i++)
        {
            v.push_back(v[i-1]+nums[i]);
        }
        return v;
    }
};
```

### 1431、拥有最多糖果的孩子

输入：candies = [2,3,5,1,3], extraCandies = 3
输出：[true,true,true,false,true] 
解释：
孩子 1 有 2 个糖果，如果他得到所有额外的糖果（3个），那么他总共有 5 个糖果，他将成为拥有最多糖果的孩子。
孩子 2 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
孩子 3 有 5 个糖果，他已经是拥有最多糖果的孩子。
孩子 4 有 1 个糖果，即使他得到所有额外的糖果，他也只有 4 个糖果，无法成为拥有糖果最多的孩子。
孩子 5 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。

```
class Solution {
public:
    vector<bool> kidsWithCandies(vector<int>& candies, int extraCandies) {
        vector<bool> v1;
        vector<int> v;
        v=candies;
        sort(candies.begin(),candies.end());
        int max=candies.back();
        for(int i=0;i<v.size();i++)
        {
            if((v[i]+extraCandies)>=max)
                v1.push_back(true);
            else
                v1.push_back(false);

        }
        return v1;
    }
};
```

### 43、字符串相乘

![image-20210808135649985](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210808135649985.png)

```c++
class Solution {
public:
    string multiply(string num1, string num2) {
        int m=num1.size();
        int n=num2.size();
        string res(m+n+1,'0');
        for(int j=0;j<n;j++)
        {
            for(int i=0;i<m;i++)
            {
                int num=(res[i+j]-'0')+(num1[m-1-i]-'0')*(num2[n-1-j]-'0');
                res[i+j]=num%10+'0';
                res[i+1+j]+=num/10;
            }
        }
        while(!res.empty()&&res.back()=='0') res.pop_back();
        return res.empty()?"0":string(res.rbegin(),res.rend());
    }
};
```



### 58、剑指Offer58-左旋转字符串II

```
class Solution {
public:
    string reverseLeftWords(string s, int n) {      
        string s1;
        s1=s.substr(0, n);
        s = s.substr(n,s.size());
        return s+s1;

    }
};

//不占用额外空间
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        reverse(s.begin(), s.begin() + n);
        reverse(s.begin() + n, s.end());
        reverse(s.begin(), s.end());
        return s;
    }
};
```

### 179、最大数

```c++
class Solution {
private:
    static bool compare(const string &a,const string &b)
    {
        return a+b>b+a;
    }
public:
    string largestNumber(vector<int>& nums) {
        vector<string> strs;
        string res;
        for(auto num:nums)
            strs.push_back(to_string(num));
        sort(strs.begin(),strs.end(),compare);
        for(auto str:strs)
            res+=str;
        if(res[0]=='0') return "0";
        return res;
    }
};
```

### 468、验证IP地址

```c++
//思路：先根据分隔符把ip地址分割成一个个的字符串
//根据ipv4或者ipv6的特点对字符串进行判断
class Solution {
public:
    vector<string> split(string s,string sign)
    {
        vector<string> res;
        int it;
        while((it=s.find(sign))&&it != s.npos)
        {
            res.push_back(s.substr(0,it));
            s=s.substr(it+1);
        }
        res.push_back(s);
        return res;
    }
    bool isIpv4(string IP)
    {
        vector<string> ss=split(IP,".");
        if(ss.size()!=4) return false;
        for(auto& st:ss)
        {
            
            if(st.size()==0||st.size()>3||(st[0]=='0'&&st.size()!=1)) return false;
            for(auto &n:st)
            {
                if(!isdigit(n)) return false;
            }
            int num=stoi(st);
            if(num>255||num<0) return false;
        }
        return true;
    }
    bool isIpv6(string IP)
    {
        vector<string> ss=split(IP,":");
        if (ss.size() != 8) return false;
        for(auto& st:ss)
        {
            if (st.size() == 0 || st.size() > 4) return false; 
            for(auto& i:st)
            {
                //必须是数字或者a-f或者A-F
                if(!(isdigit(i)||(i>='a'&&i<='f')||(i>='A'&&i<='F')))
                    return false;
            }
        }
        
        return true;
    }
    string validIPAddress(string IP) {
        if(isIpv4(IP))
        {
            return "IPv4";
        }
        else if(isIpv6(IP))
        {
            return "IPv6";
        }
        else return "Neither";
    }
};
```



### 剑指 31、奇数位于偶数的前面（双指针）

```c++
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int n=nums.size();
        int left=0;
        int right=n-1;
        while(left<right)
        {
            if(nums[left]%2!=0&&nums[right]%2==0)
            {
                left++;
                right--;
            }
            else if(nums[left]%2==0&&nums[right]%2==0)
            {
                right--;
            }
            else if(nums[left]%2!=0&&nums[right]%2!=0)
            {
                left++;
            }
            else{
                swap(nums[left],nums[right]);
            }
        }
        return nums;
    }
};
```

### 498、对角线遍历

### ![image-20210810162621006](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210810162621006.png)

```c++
/*
向右上遍历时
判断顺序不可颠倒，因为在遍历到右上角元素时，先判断“第一行”的话下标会溢出

元素在最后一列，向下走
元素在第一行，向右走
向左下遍历时

元素在最后一行，向右
元素在第一列，向下
其余情况就是正常的向右上或者左下遍历
*/
class Solution 
{
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) 
    {
        //判空
        if (matrix.size() == 0) return {};
        int m = matrix.size();
        int n = matrix[0].size();
        int count = m * n;
        vector<int> ans;
        int x = 0;
        int y = 0;
        for (int i = 0; i < count; ++i)
        {
            ans.push_back(matrix[x][y]);
            if ((x+y) % 2 == 0)
            {
                if (y == n-1)   //最后一列，向下
                {
                    x++;
                }else if (x == 0)//第一行，向右
                {
                    y++;
                }else           //向右上
                {
                    x--;
                    y++;
                }
            }else
            {
                if (x == m-1)   //最后一行，向右
                {
                    y++;
                }else if (y == 0)//第一列，向下
                {
                    x++;
                }else           //向左下
                {
                    x++;
                    y--;
                }
            }
        }
        return ans;
    }
};
 
```

### 670、最大交换（贪心）

```c++
//找到尽可能高位的一个数，该数满足其右边的低位的最大值比其大，将这两个数交换，如果有多位为最大值，取最低位。
//1、首先找到真个数字中，从高位到最低位严格递增的第一个位置t,即s[t-1]<s[t]，若不存在这样的t,则不需要交换
//2、从t之后再找有没有最大值，有的话定位到最大值的位置，注意如果有相同的最大值，需要定位到最低位
//3、从最高位开始，进行交换
class Solution {
public:
    int maximumSwap(int num) {
        string s=to_string(num);
        int l=s.length(),t=-1;
        for(int i=0;i<l-1;i++)
        {
            if(s[i]<s[i+1])
            {
                t=i+1;
                break;
            }
        }
        if(t==-1) return num;
        //找到i以后的最大值的位置下标
        int maxi=t;
        for(int i=t+1;i<l;i++)
        {
            if(s[maxi]<=s[i])  //注意这里是<=,保证相等的情况下取到最低位
                maxi=i;
        }
        for(int i=0;i<l;i++)
        {
            if(s[i]<s[maxi])
            {
                swap(s[i],s[maxi]);
                break;
            }
        }
        int ans=0;
        for(int i=0;i<l;i++)
        {
            ans=ans*10+s[i]-'0';
        }
        return ans;
    }
};
```

### 207、课程表（拓扑排序）

![image-20210825165715787](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210825165715787.png)

![image-20210825172915072](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210825172915072.png)

```c++
/*
问题可以简化为：课程安排图是否是 有向无环图(DAG)。
拓扑排序原理： 对 DAG 的顶点进行排序，使得对每一条有向边 (u,v)，均有 u（在排序记录中）比 v 先出现。亦可理解为对某点 v 而言，只有当 v 的所有源点均出现了，v 才能出现。
*/
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> degrees(numCourses);
        vector<vector<int>> adjacents(numCourses);
        queue<int> zero;
        int num=numCourses;
        for(int i=0;i<prerequisites.size();i++)
        {
            degrees[prerequisites[i][0]]++;
            adjacents[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        for(int i=0;i<numCourses;i++)
        {
            if(degrees[i]==0)
            {
                zero.push(i);
                num--;
            }
        }
        while(!zero.empty())
        {
            int tmp=zero.front();
            zero.pop();
            for(int j=0;j<adjacents[tmp].size();j++)
            {
                if(--degrees[adjacents[tmp][j]]==0)
                {
                    zero.push(adjacents[tmp][j]);
                    num--;
                }
            }
        }
        if(num==0) return true;
        return false;

    }
};
```



### 887、鸡蛋掉落

```c++
/*
0、N 和 F 的关系：F 比 N 多一个 0 层
1、问题转换：N 个楼层，有 K 个蛋，求最少要扔 T 次才能确定 F 的值--->有 K 个蛋，扔 T 次，求可以确定 F 的个数，然后得出 N 个楼层
2、如果只有 1 个蛋：只能从低层开始仍；有 T 次机会，只可以确定出 T + 1 个 F
3、如果只有 1 次机会：等同于只有 1 个蛋，同2
4、计算能确定 F 的个数
	如果只有 1 个蛋，或只有 1 次机会时，只可以确定出 T + 1 个 F
	其他情况时，递归。【蛋碎了减 1 个，机会减 1 次】 + 【蛋没碎，机会减 1 次】
5、题目给出了 K ，不断增大 T ，计算出来的 F 的个数已经超过了 N + 1 时，就找到了答案 T代表机会
*/
class Solution {
public:
    //返回的是f的个数
    int calf(int k,int t)
    {
        if(k==1||t==1) return t+1;
        return calf(k-1,t-1)+calf(k,t-1);  //碎+没有碎的情况
    }
    int superEggDrop(int k, int n) {
        int t=1;
        while(calf(k,t)<n+1) t++;
        return t;
    }
};
```

### 36、有效的数独（位运算）

![image-20210826141559197](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210826141559197.png)

```c++
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        int row[9][10]={0};  //第几行哪个数，之所以是10是因为数字是1-9，下标为9开辟的空间就是10
        int col[9][10]={0};  //第几列哪个数
        int box[9][10] = {0};  //第几个box
        for(int i=0;i<9;i++)
        {
            for(int j=0;j<9;j++)
            {
                if(board[i][j]=='.') continue;
                int curNumber = board[i][j]-'0';
                if(row[i][curNumber]) return false; 
                if(col[j][curNumber]) return false;
                if(box[j/3 + (i/3)*3][curNumber]) return false;

                row[i][curNumber] = 1;
                col[j][curNumber]=1;
                box[j/3 + (i/3)*3][curNumber]=1;
            }
        }
        return true;
    }
};
```



## 脑筋急转弯

### 292、Nim游戏

```
class Solution {
public:
    bool canWinNim(int n) {
        return n%4!=0;
    }
};
```

### 877、石子游戏

```
class Solution {
public:
    bool stoneGame(vector<int>& piles) {
        return true;
    }
};
//动态规划解决

```

### 319、灯泡开关

```
class Solution {
public:
    int bulbSwitch(int n) {
        return (int)sqrt(n);
    }
};
```

## 剑指offer

### 51、数组中的逆序对（归并排序）

![Picture2.png](https://pic.leetcode-cn.com/1614274007-rtFHbG-Picture2.png)

```c++
class Solution {
public:
    int ret = 0;
    vector<int> tmp;
    
    void merge(vector<int>& nums, int l, int r){
        int mid = l + (r - l) / 2;
        int pos1 = l, pos2 = mid + 1, p = l;
        while(pos1 <= mid && pos2 <= r){
            if(nums[pos1] <= nums[pos2]) tmp[p++] = nums[pos1++];
            else{
                tmp[p++] = nums[pos2++];
                ret += (mid - pos1 + 1);
            }
        }
        while(pos1 <= mid) tmp[p++] = nums[pos1++];
        while(pos2 <= r) tmp[p++] = nums[pos2++];
        for(int i = l; i <= r; ++i) nums[i] = tmp[i];
    }

    void sort(vector<int>& nums, int l, int r){
        if(l >= r) return;
        int mid = l + (r - l) / 2;
        sort(nums, l, mid);
        sort(nums, mid + 1, r);
        merge(nums, l, r);
    }

    int reversePairs(vector<int>& nums) {
        tmp.resize(nums.size());    
        sort(nums, 0, nums.size() - 1);
        return ret;
    }
};
```

### 912、排序数组

```c++
//归并排序
class Solution {
public:
    vector<int> tmp;
    void merge(vector<int>& nums,int l,int r)
    {
        if(l >= r) return;
        int mid=l+(r-l)/2;
        merge(nums,l,mid);
        merge(nums,mid+1,r);
        int pos1=l;
        int pos2=mid+1;
        int p=l;
        while(pos1<=mid&&pos2<=r)
        {
            if(nums[pos1]<=nums[pos2]) 
            {
                tmp[p++]=nums[pos1++];
            }
            else
            {
                tmp[p++]=nums[pos2++];
            }
        }
        while(pos1<=mid) tmp[p++]=nums[pos1++];
        while(pos2<=r) tmp[p++]=nums[pos2++];
        for(int i=l;i<=r;i++) nums[i]=tmp[i];
    }
    vector<int> sortArray(vector<int>& nums) {
        tmp.resize(nums.size());
        merge(nums,0,nums.size()-1);
        return tmp;
    }
};
//快速排序
//当元素少的时候用插入
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;        
// 优化1 : 随机打乱
        random_shuffle(nums.begin(), nums.end());
        quick_sort(nums, l, r);
        return nums;
    }
    void quick_sort(vector<int>& nums, int left, int right) {
        if (left >= right) return;
// 优化2 : 当元素小于等于5个的时候用插入排序
        else if (right - left < 5) {
            insertion_sort(nums, left, right);
            return;
        }
        int l = left, r = right + 1;
        int v = nums[left];
        while (true) {
            while (nums[++l] < v) if (l == right) break;
            while (nums[--r] > v);
            if (l >= r) break;
            swap(nums[l], nums[r]);
        }
        swap(nums[r], nums[left]);
        quick_sort(nums, left, r - 1);
        quick_sort(nums, r + 1, right);
    }
    void insertion_sort(vector<int>& nums, int left, int right) {
        for (int i = left + 1; i <= right; ++i) {
            for (int j = i; j > left && nums[j] < nums[j - 1]; --j) {
                swap(nums[j], nums[j - 1]);
            }
        }
    }
};
 
```



### 53、0~n-1中缺失的数字（暴力、位运算）

```c++
//暴力
class Solution {
public:
    int missingNumber(vector<int>& nums) {
           
        int length = nums.size();
        for (int i = 0; i < length; i++) {
            if (nums[i] != i)
                return i;
        }
        return length;
    }
 
    
};
//位运算
/*
当nums[mid]==mid: 证明在右边
否则在左边
*/
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int left=0;
        int right=nums.size()-1;
        while(left<=right)
        {
            int mid=left+(right-left)/2;
            if(nums[mid]==mid)
            {
                left=mid+1;
            }
            else
            {
                right=mid-1;
            }
        }
        return left;
        
    }
};
//yxc
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int left=0;
        int right=nums.size();
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]>mid) right=mid;
            else left=mid+1;
        }
        return left;
    }
};
```

### 33、二叉搜索树的后序遍历序列（递归）

```c++
class Solution {
public:
    bool VerifySquenceOfBSTCore(vector<int>& sequence, int start, int end) {
    if (start >= end) return true;
    int low = start;
    while (low < end && sequence[low] < sequence[end])  ++low;

    for (int i = low; i < end; ++i) {
        if (sequence[i] <= sequence[end]) return false;
    }

    return  VerifySquenceOfBSTCore(sequence, start,low-1) &&
        VerifySquenceOfBSTCore(sequence, low,end-1);
}

    bool verifyPostorder(vector<int>& postorder) {
        if (postorder.empty())  return true;
    if (postorder.size() == 1) return true;
    return VerifySquenceOfBSTCore(postorder, 0, postorder.size()-1);
    }
};
```

### 65、不用加减乘除做加法（位运算）

```
class Solution {
public:
    int add(int a, int b) {
        if(b==0)
            return a;
        return add(a^b,(unsigned int)(a&b)<<1);
    }
};
```

### 46、把数字翻译成字符串（DP）

```c++
//对于下标为I的情况：如果不能与前面的组合就是和dp[i-1]一样，如果能组合就增加了一种dp[i-2]
class Solution {
public:
    int translateNum(int num) {
        string s="0"+to_string(num);
        vector<int> dp(s.size(),0);
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<s.size();i++)
        {
            dp[i]=dp[i-1];
            int tmp=stoi(s.substr(i-1,2));
            if(tmp>=10&&tmp<=25)
            {
                dp[i]+=dp[i-2];
            }
        }
        return dp[s.size()-1];

    }
};

//简化版
class Solution {
public:
    int translateNum(int num) {
        string s=to_string(num);
        int pre=1;
        int pree=1;
        int tmp;
        for(int i=2;i<=s.size();i++)
        {
            int a=stoi(s.substr(i-2,2));  //从哪里截取，截取多长
            if(a>=10&&a<=25)
            {
                  tmp=pre+pree;
            }
            else
            {
                  tmp=pre;
            }
            pree=pre;
            pre=tmp;
            
        }
        return pre;
    }
};
```

### 11、旋转数组的最小数字（二分）

```c++
class Solution {
public:
    int minArray(vector<int>& numbers) {
        sort(numbers.begin(),numbers.end());
        return numbers[0];
    }
};
//二分查找
class Solution {
public:
    int minArray(vector<int>& numbers) {
        int i=0;
        int j=numbers.size()-1;
        while(i<j)
        {
            int m=(i+j)/2;
            if(numbers[m]<numbers[j])
            {
                j=m;
            }
            else if(numbers[m]>numbers[j])
            {
                i=m+1;
            }
            else
                j--;
        }
        return numbers[i];
    }
};
```

### 162、寻找峰值(二分)

```c++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int l=0,r=nums.size()-1;
        while(l<r)
        {
            int mid=(l+r)/2;
            if(nums[mid]>nums[mid+1]) r=mid;
            else l=mid+1;

        }
        return r;
    }
};
```

### 1095、山脉数组中查找目标值（二分）

```c++
/*
1、通过三分查找定位山的最高点，然后再通过二分在左右山峰查找目标值
*/

class Solution {
public:
    int getHighest(int n,MountainArray &mountainArr)
    {
        int l=0,r=n-1;
        while(l+2<r)
        {
            int mid1=l+(r-l)/3;
            int mid2=r-(r-l)/3;
            if(mountainArr.get(mid1)>mountainArr.get(mid2))
                r=mid2;
            else
                l=mid1;
        }
        //剩下最后三个数用暴力
        if(l==r)
            return l;
        else if(l+1==r)
        {
            if(mountainArr.get(l)>mountainArr.get(l+1))
                return l;
            else return r;
        }
        else{
            int x=mountainArr.get(l);
            int y=mountainArr.get(l+1);
            int z=mountainArr.get(l+2);
            if(x>y&&x>z) return l;
            else if(y>x&&y>z) return l+1;
            else return l+2;
        }
    }
    int findInMountainArray(int target, MountainArray &mountainArr) {
        int n=mountainArr.length();
        int high=getHighest(n,mountainArr);
        int l=0,r=high;
        while(l<r)
        {
            int mid=(l+r)/2;
            if(mountainArr.get(mid)>=target) r=mid;
            else l=mid+1;
        }
        if(mountainArr.get(r)==target) return l;
        l=high-1;
        r=n-1;
        while(l<r)
        {
            int mid=(l+r)/2;
            if(target>=mountainArr.get(mid)) r=mid;
            else l=mid+1;
        }
        if(mountainArr.get(r)==target) return l;
        else return -1;
    }
};
```

### 704、二分查找

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left=0;
        int right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right+1)/2;
            if(nums[mid]<=target) left=mid;
            else right=mid-1;
        }
        if(nums[left]==target) return left;
        return -1;
    }
};
```

### 59、队列的最大值

```c++
class MaxQueue {
private:
    queue<int> que;
    deque<int> deq;
public:
    MaxQueue() {

    }
    
    int max_value() {
        if(deq.size() == 0)
            return -1;
        return deq.front();
    }
    
    void push_back(int value) {
        que.push(value);
        while(!deq.empty()&&value>deq.back())
        {
            deq.pop_back();
        }
        deq.push_back(value);
        
    }
    j
    int pop_front() {
        if(que.size() == 0)
            return -1;
        int res=que.front();
        que.pop();
        if(res==deq.front())
            deq.pop_front();
        return res;
    }
}; 
```

### 29(54)、螺旋矩阵

```c++
class Solution 
{
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) 
    {
        if (matrix.empty()) return {};
        vector<int> res;
        int l = 0;                      //左边界
        int r = matrix[0].size() - 1;   //右边界
        int t = 0;                      //上边界
        int b = matrix.size() - 1;      //下边界
        while (true)
        {
            //left -> right
            for (int i = l; i <= r; i++) res.push_back(matrix[t][i]);
            if (++t > b) break;
            //top -> bottom
            for (int i = t; i <= b; i++) res.push_back(matrix[i][r]);
            if (--r < l) break;
            //right -> left
            for (int i = r; i >= l; i--) res.push_back(matrix[b][i]);
            if (--b < t) break;
            //bottom -> top
            for (int i = b; i >= t; i--) res.push_back(matrix[i][l]);
            if (++l > r) break;
        }
        return res;
    }
};
```

### 59、螺旋矩阵||

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
	vector<vector<int>> res(n, vector<int>(n, 0));
	int top = 0;
	int right = n-1 ;
	int bottom = n-1;
	int left = 0;
	int num = 1;
	int end = n * n;
	while (num <= end)
	{
		for (int i = left; i <= right; i++)		res[top][i] = num++;	//i代表列---从左到右
		top++;

		for (int i = top; i <= bottom; i++)		res[i][right] = num++;	//i代表行---从上到下
		right--;

		for (int i = right; i >= left; i--)		res[bottom][i] = num++;	//i代表列---从右到左
		bottom--;

		for (int i = bottom; i >= top; i--)		res[i][left] = num++;	//i代表行---从下到上
		left++;
	}
	return res;
    }
};
```



### 35、复杂链表的复制

```
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head==NULL) return NULL;
        unordered_map<Node *,Node *> map;
        Node *cur=head;
        while(cur!=NULL)
        {
            map[cur]=new Node(cur->val);
            cur=cur->next;
        }
        cur=head;
        while(cur!=NULL)
        {
            map[cur]->next=map[cur->next];
            map[cur]->random=map[cur->random];
            cur=cur->next;
        }
        return map[head];
    }
};
```

### 61、扑克牌中的顺子（set）

```c++
class Solution {
public:
    //除去0之外有重复的数直接返回
    bool isStraight(vector<int>& nums) {
        set<int> se;
        for(auto num:nums)
        {
            if(num==0) continue;
            if(se.find(num)!=se.end())   // 若有重复，提前返回 false
                return 0;
            se.insert(num);
        }
        return (*se.rbegin()-*se.begin())<5;  //指向最后一个数据的迭代器
    }
};

```



### 36、二叉搜索树与双向链表

```c++
/*
dfs(cur): 递归法中序遍历；
1、终止条件： 当节点 cur 为空，代表越过叶节点，直接返回；
2、递归左子树，即 dfs(cur.left) ；
3、构建链表：
	当 pre 为空时： 代表正在访问链表头节点，记为 head ；
	当 pre 不为空时： 修改双向节点引用，即 pre.right = cur ， cur.left = pre ；
	保存 cur ： 更新 pre = cur ，即节点 cur 是后继节点的 pre ；
4、递归右子树，即 dfs(cur.right) ；
*/
class Solution {
private:
    Node* pre,*head;
    void dfs(Node* cur) {   //因为是中序遍历，所以cur指针会按照从小到大的顺序移动
        if(cur == nullptr) return;
        dfs(cur->left);
        if(pre != nullptr) pre->right = cur;
        else head = cur;
        cur->left = pre;
        pre = cur;
        dfs(cur->right);
    }
     
public:
    Node* treeToDoublyList(Node* root) {
        
        if(root == nullptr) return nullptr;
        dfs(root);
        //dfs结束后pre现在指向的是最后一个节点
        head->left = pre;
        pre->right = head;
        return head;
    }
};
```

### 31、栈的压入弹出序列

```
class Solution {
public:
    bool validateStackSequences(vector<int>& v1, vector<int>& v2) {
        stack<int> st;
        int i=0;
        for(auto num:v1)
        {
            st.push(num);
            while(!st.empty()&&st.top()==v2[i])
            {
                st.pop();
                i++;
            }
        }
        return st.empty();
    }
};
```

### 53、在排序数组中查找数字 |

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        unordered_map<int,int> map;
        for(auto num:nums)
        {
            map[num]++;
        }
        for(auto [k,v]:map)
        {
            if(k==target)
                return v;
        }
        return 0;
    }
};
//二分
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.empty()) return 0;
        int left=0;
        int right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]>=target) right=mid;
            else left=mid+1;
        }
        if(nums[right]!=target) return 0;
        int start=right;

         left=0;
         right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right+1)/2;
            if(nums[mid]<=target) left=mid;
            else right=mid-1;
        }
        int end=right;
        return end-start+1;
    }
};
```

### 45、把数组排成最小的数

```c++
//使用lambda表达式
class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string>strs;
        string ans;
        for(int i = 0; i < nums.size(); i ++){
            strs.push_back(to_string(nums[i]));
        }
        sort(strs.begin(), strs.end(), [](string& s1, string& s2){return s1 + s2 < s2 + s1;});
        for(int i = 0; i < strs.size(); i ++)
            ans += strs[i];
        return ans;
    }
};


class Solution {
public:
    string minNumber(vector<int>& nums) {
        vector<string> strs;
        string res;
        for(auto num:nums)
            strs.push_back(to_string(num));
        sort(strs.begin(),strs.end(),compare);
        for(auto str:strs)
            res+=str;
        return res;
    }
private:
    static bool compare(const string &a,const string &b)
    {
        return a+b<b+a;
    }
};
 
```

### 26、树的子结构

```c++
class Solution {
public:
    //以A为根是否包含B
    bool recur(TreeNode* A,TreeNode* B)
    {
        if(B==NULL)
            return true;
        if(A==NULL)
            return false;
        if(A->val==B->val)
            return recur(A->left,B->left)&&recur(A->right,B->right);
        return false;
    }
    //A树是否包含B树
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if (B == NULL || A == NULL) {
            return false;
        }
        if(A->val==B->val && (recur(A->left,B->left)&& recur(A->right, B->right)))
        {
            return true;
        }
        return isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }
};
```

### 37、序列化二叉树

序列化：

空用x, 其他用数字
用，来分隔； 最后一个也加上，分割时候会忽略
反序列化：

先把string按照，来分割生产queue
queue来保存当前的要遍历的结点，每次遍历当前queue.size()的层的每个结点
每个子结点从左到右的顺序， 只有非空，才会进入queue，用于下次遍历

```c++
class Codec {
public:
	string serialize(TreeNode* root) {

		string r;
		queue<TreeNode*> q;

		q.push(root);

		while (!q.empty())
        {
			auto c = q.front();
			if (c != nullptr)
            {
				q.push(c->left);
				q.push(c->right);
			}

			q.pop();

			if (c == nullptr)
            {
				r.append("x");
			}
			else
            {
				r.append(to_string(c->val));
			}

			r.append(",");
		}
		return r;
	}

    // 从字符串s里按照，拆分出每个结果插入到队列里 
    void Transfer(queue<string>& v, const string& s)
	{
		int begin = 0;
        int n = s.size();
        int start = 0;
        for (int i = 0; i < n; ++i)
        {
            if (s[i] == ',')
            {
                v.emplace(s.substr(start, i-start));
                start = i+1;
            }
        }
        //  默认总是有一个 , 所以无需担心结束情况
	}
    
	TreeNode* make(queue<string> & v)
	{
		TreeNode* root = nullptr;

		queue<TreeNode*> q;

        // 设置第一个根结点
		string head = v.front();
		if (head != "x") {
			root = new TreeNode(atoi(head.c_str()));
			q.push(root);		
		}
		v.pop();

        int currSize;
		while (!v.empty())
		{   
            currSize = q.size();
            for (int i = 0; i < currSize; ++i)
            {
                TreeNode* curr = q.front();
                q.pop();
                string currLeft = v.front();
                v.pop();
                string currRight = v.front();
                v.pop();
                if (currLeft != "x")
                {
                    TreeNode* left = new TreeNode(std::stoi(currLeft));
                    curr->left = left;
                    q.push(left);
                }
                if (currRight != "x")
                {
                    TreeNode* right = new TreeNode(std::stoi(currRight));
                    curr->right = right;
                    q.push(right);
                }
            }
		}

		return root;
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data)
    {
		queue<string> v;
		Transfer(v, data);
		return make(v);
	}
};
```

### 60、N个骰子的点数（DP）

```c++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> res(n * 6 - n + 1);
        vector<vector<int> > dp(n + 1, vector<int>(6 * n + 1, 0)); // 将全部值初始化为 0
        int row = n+1, col = 6*n+1; //col ==> [0,6n] 共6n+1列
        // 初始化第一个骰子
        // dp[1][1]=1 代表一个骰子点数和为 1 的情况有一种
        // dp[1][2]=1 代表一个骰子点数和为 2 的情况有一种
        for(int i=1;i <= 6;i++){
            dp[1][i] = 1; //因为只有一个骰子时，点数和为1,2,3,4,5,6的情况都各只有一种
        }

        for(int i=2;i < row;i++){ //从2颗骰子开始计算
            for(int j=i;j < col;j++){ //j就是点数之和s,是从i开始的，
                                      //代表i个骰子点数之和的最小值为i
                for(int k=1;k <= 6;k++){
                //dp[i][j]表示 i个骰子点数之和为j的数量有多少？
                //答：当第i个骰子点数为k时，那么前i-1个骰子点数和必须等于j-k，则k+(j-k)=j;
                //    把所有dp[i-1][j-k]的数量加起来(k∈[1,6])，就等于dp[i][j]的数量了
                   if(j-k > 0)  dp[i][j] += dp[i-1][j-k];
                   else break;
                }
            }
        }

        double denominator = pow(6.0, n); // 分母
        int index=0;
        for(int k = n;k <= 6*n;k++){ //计算概率 放入答案
           res[index++] = dp[n][k] / denominator; //n个骰子，共可以产生6^n种结果
                                               //比如，2个骰子可以产生C(1,6)*C(1,6)=6*6种
        }
        return res;
    }
};
```

### 43、1~n整数中1出现的次数

```c++
//解题思路：当cur=0时，high×digit；当cur=1时：high×digit+low+1；当cur=2,3,⋯,9 :(high+1)×digit
 class Solution {
public:
    int countDigitOne(int n) {
        int hight=n/10;
        int cur=n%10;
        long dight=1;
        int low=0;
        int res=0;  //存放结果
        while(hight!=0||cur!=0)
        {
            if(cur==0)
            {
                res+=hight*dight;
            }
            else if(cur==1)
            {
                res+=hight*dight+1+low;
            }
            else{
                res+=(hight+1)*dight;
            }
            //移动
            low += cur * dight;
            cur = hight % 10;
            hight/=10;
            dight*=10;

        }
        return res;
    }
};

```

### 10、青蛙跳台阶问题

```c++
class Solution {
public:
    int numWays(int n) {
        int a = 1, b = 1, sum;
        for(int i = 0; i < n; i++){
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
 
    }
};
```

### 39、数组中出现次数超过半数的数字

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int champin=nums[0];
        int count=1;
        for(int i=1;i<nums.size();i++)
        {
            if(champin!=nums[i])
            {
                count--;
                if(count==0)
                    champin=nums[i+1];
            }
            else
            {
                count++;
            }
        }
        return champin;
    }
};
```

### 384、打乱数组

![image-20210812145441682](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210812145441682.png)

```c++
/*
洗牌算法（Knuth shuffle算法）：对于有n个元素的数组来说，为了保证洗牌的公平性，应该要能够等概率的洗出n!种结果。
举例解释如下：

开始数组中有五个元素；
在前五个数中随机选一个数与第五个数进行交换，每个数都有五分之一的概率被交换到最后一个位置；
在前四个数中随机选一个数与第四个数进行交换，每个数都有五分之一的概率被交换到第四个位置；
在前三个数中随机选一个数与第三个数进行交换，每个数都有五分之一的概率被交换到第三个位置；

*/
class Solution {
private:
    vector<int> original;
public:
    Solution(vector<int>& nums) {
        original=nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return original;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> nums(original);
        for(int i=nums.size()-1;i>=0;i--)
        {
            swap(nums[i], nums[rand() % (i + 1)]);//rand()%(i+1)能随机生成0到i中的任意整数
        }
        return nums;
    }
};
 
```



## 自己做的

### 面试04.02、最小高度树

```
class Solution {
public:
    TreeNode* digui(vector<int>& nums,int left,int right)
    {
        if(left>right)
            return NULL;
        int n=(right+left)/2;
       
        TreeNode *root=new TreeNode(nums[n]);
        root->left=digui(nums,left,n-1);
        root->right=digui(nums,n+1,right);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {

        return digui(nums,0,nums.size()-1);
    }
};
```

### 面试17.16、按摩师

```
class Solution {
public:
    int massage(vector<int>& nums) {
        vector<int> dp(nums.size()+1,0);
        if(nums.size()==0) return 0;
        if(nums.size()==1) return nums[0];
        if(nums.size()==2) return max(nums[0],nums[1]);
        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        for(int i=2;i<nums.size();i++)
        {
            dp[i]=max(dp[i-2]+nums[i],dp[i-1]);
        }
        return dp[nums.size()-1];
    }
};
```

### 面试02.03、删除中间节点

```
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val=node->next->val;
        node->next=node->next->next;
    }
};
```

### 169、多数元素

```c++
//哈希法
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> umap;
        int n=(nums.size())/2;
        for(int a:nums)
        {
            umap[a]++;
            if(umap[a]>n)
            {
                return a;
            }
        }
        return 0;

    }
};
//排序
class Solution {
public:
    int majorityElement(vector<int>& nums) {
       sort(nums.begin(),nums.end());
       return nums[nums.size()/2];

    }
};
//摩尔投票法
/*
玩一个诸侯争霸的游戏，假设你方人口超过总人口一半以上，并且能保证每个人口出去干仗都能一对一同归于尽。最后还有人活下来的国家就是胜利。那就大混战呗，最差所有人都联合起来对付你（对应你每次选择作为计数器的数都是众数），或者其他国家也会相互攻击（会选择其他数作为计数器的数），但是只要你们不要内斗，最后肯定你赢。最后能剩下的必定是自己人。
*/
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int count=0;
        int condi=nums[0];
        for(auto n:nums)
        {
            if(count==0) condi=n;
            if(condi==n)  count++;
            else count--;
        }
        return condi;
    }
};
```

### 283、移动零（双指针）

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
      int slow=0;
      int fast=0;
      int count=0;
      int size=nums.size();
      while(fast<size)
      {
          if(nums[fast]!=0)
          {
              nums[slow]=nums[fast];
              slow++;
              count++;
          }
          fast++;
      } 
      int m=size-count;
      for(int i=size-1;i>=count;i--)
      {
          nums[i]=0;
      } 
    }
};
class Solution {
public:
    //一次遍历
    void moveZeroes(vector<int>& nums) {
        if(nums.size()==0) return;
        int j=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]!=0)
            {
                int tmp=nums[i];
                nums[i]=nums[j];
                nums[j]=tmp;
                j++;
            }
        }
    }
};
```

### 22、括号生成

```c++
//回溯法
//考虑什么时候括号合法：左括号和右括号的个数大小的时候
class Solution {
public:
    vector<string> result;
    string path;
    void backtracking(int n,int open_num,int close_num)
    {
        if(path.size()==2*n)
        {
            result.push_back(path);
            return;
        }
        //只有这两种情况才是合法的括号
        if(open_num<n)
        {
            path.push_back('(');
            backtracking(n,open_num+1,close_num);
            //不能写成backtracking(n,open_num+1,close_num);
            path.pop_back();
        }
        if(close_num<open_num)
        {
            path.push_back(')');
            backtracking(n,open_num,close_num+1);
            path.pop_back();
        }
    }
    vector<string> generateParenthesis(int n) {
        backtracking(n,0,0);
        return result;
    }
};
```

### 26、删除排序数组的重复项（双指针）

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if(nums.size()==0) return 0;

        int slow=0;
        int fast=1;
        while(fast<nums.size())
        {
            if(nums[slow]!=nums[fast])
            {
                nums[slow+1]=nums[fast];
                slow++;
            }
            fast++;
        }
        return slow+1;
    }
};
```

### 2、两数相加

```c++
//获取长度、补成长度一样长的、相加
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int len1=0;
        int len2=0;
        ListNode* p=l1;
        ListNode* q=l2;
        while(p->next!=NULL)
        {
            len1++;
            p=p->next;
        }
        while(q->next!=NULL)
        {
            len2++;
            q=q->next;
        }
        if(len1>len2)
        {
            for(int i=1;i<=len1-len2;i++)
            {
                q->next=new ListNode(0);
                q=q->next;
            }
        }
        else//l2较长，在l1末尾补零
        {
            for(int i=1;i<=len2-len1;i++)
            {
                p->next=new ListNode(0);
                p=p->next;
            }
        }
        p=l1;
        q=l2;
        int flag=0;  //进位标志
        ListNode* sum_node=new ListNode(-1);
        ListNode*w=sum_node;
        int sum=0;
        while(p!=NULL)
        {
            sum=flag+p->val+q->val;
            w->next=new ListNode(sum%10);
            flag=(sum>=10)?1:0;
            w=w->next;
            p=p->next;
            q=q->next;
        }
        if(flag)
        {
            w->next=new ListNode(1);
            w=w->next;
        }
        return sum_node->next;

    }
};
```

### 67、二进制求和

```
class Solution {
public:
    string addBinary(string a, string b) {
        //判断字符串的长度
        int al=a.size();
        int bl=b.size();
        while(al<bl)
        {
            a='0'+a;
            ++al;
        }
        while(al>bl)
        {
            b='0'+b;
            ++bl;
        }
        for(int j=a.size()-1;j>0;j--)
        {
            a[j]=a[j]-'0'+b[j];
            if(a[j]>='2')
            {
                a[j]=(a[j]-'0')%2+'0';
                a[j-1]=a[j-1]+1;
            }
        }
        //单独处理第一位
        a[0]=a[0]-'0'+b[0];
        if(a[0]>='2')
        {
            a[0]=(a[0]-'0')%2+'0';
            a='1'+a;
        }
        return a;
    }
};
```



### 3、无重复字符的最长字串(set、滑动窗口)

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size()==0) return 0;
        int left=0;  //滑动窗口的左边界
        int maxleng=0;
        unordered_set<char> lookup;
        for(int i=0;i<s.size();i++)
        {
            while(lookup.find(s[i])!=lookup.end())   //找到了
            {
                lookup.erase(s[left]);
                left++;
            }
            maxleng=max(maxleng,i-left+1);
            lookup.insert(s[i]);

        }
        return maxleng;
    }
};
```

### 448、找到所有数组中消失的数字

```
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n=nums.size();
        vector<int> result;
        unordered_set<int> lookup;
        for(int i=0;i<n;i++)
        {
            lookup.insert(nums[i]);
        }
        for(int i=1;i<=n;i++)
        {
            if(lookup.find(i)==lookup.end())
            {
                result.push_back(i);
            }
        }
        return result;
    }
};
```

### 41、缺失的第一个正数（原地哈希）

```c++
//思路：
//1、缺失的数字肯定是在1~n+1之间
//2、首先遍历一遍数组把位于1~n之间的数放到原数组的对应的位置
//3、再遍历一遍数组，不满足条件的返回下标i+1
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n=nums.size();
        for(int i=0;i<n;i++)
        {
            while(nums[i]>=1&&nums[i]<=n&&(nums[i]!=nums[nums[i]-1]))
                swap(nums[i],nums[nums[i]-1]);
        }
        int i;
        for(i=0;i<n;i++)
        {
            if(nums[i]!=i+1)
                return i+1;
        }
        return i+1;
    }
};
```



### 46、全排列

```c++
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& nums,vector<bool>&used)
    {
        if(path.size()==nums.size())
        {
            result.push_back(path);
            return;
        }
        for(int i=0;i<nums.size();i++)
        {
            if(used[i]==true)
            {
                continue;
            }
            used[i]=true;
            path.push_back(nums[i]);
            backtracking(nums,used);
            path.pop_back();
            used[i]=false;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        backtracking(nums, used);
        return result;
    }
};
//yxc
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;
    int n;
    vector<bool> st;
    vector<vector<int>> permute(vector<int>& nums) {
        n=nums.size();
        st=vector<bool>(n);
        dfs(nums,0);
        return ans;        
    }
    void dfs(vector<int>& nums,int u)
    {
        if(u==n)
        {
            ans.push_back(path);
            return;
        }
        for(int i=0;i<n;i++)
        {
            if(!st[i])
            {
                st[i]=true;
                path.push_back(nums[i]);
                dfs(nums,u+1);
                path.pop_back();
                st[i]=false;
            }
        }
    }
};
```

### 5、最长回文子串（动态规划）

回文串：不会因为读的前后顺序而不同

```c++
//解题思路
/*
* dp[i][j]表示s[i....j]是否是回文串
* 当j-i<3时直接判断就可以
* dp[i][j]=s[i]==s[j]&&dp[i+1][j-1];
*/
class Solution {
public:
    string longestPalindrome(string s) {
        int length=s.size();
        
        int begin=0;
        if(length<2)
            return s;
        vector<vector<int>> dp(length+1,vector<int>(length+1,0));
        for(int i=0;i<length;i++)
        {
            dp[i][i]=1;
           
        }
        int maxlen=1;
        for(int j=1;j<length;j++)
        {
            for(int i=0;i<j;i++)
            {
                if(s[i]!=s[j])
                {
                    dp[i][j]=0;
                }
                else
                {
                    if(j-i<3)
                    {
                        dp[i][j]=1;
                    }
                    else{
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if(dp[i][j]==1&&j-i+1>maxlen)
                {
                    maxlen=j-i+1;
                    begin=i;
                }
            }
        }
        return s.substr(begin,maxlen);
        
    }
};
//字符串方法
//每次找中心点  要注意奇偶数量
class Solution {
public:
    string longestPalindrome(string s) {
        string res;
        for(int i=0;i<s.size();i++)
        {
            for(int j=i,k=i;j>=0&&k<s.size()&&s[j]==s[k];j--,k++)
            {
                if(res.size()<k-j+1)
                {
                    res=s.substr(j,k-j+1);
                }
            }
            for(int j=i,k=i+1;j>=0&&k<s.size()&&s[j]==s[k];j--,k++)
            {
                if(res.size()<k-j+1)
                {
                    res=s.substr(j,k-j+1);
                }
            }
        }
        return res;
    }
};
```

### 647、回文串

```c++
//思路：先把二维表格填充好，再统计1的个数
class Solution {
public:
    int countSubstrings(string s) {
        int n=s.size();
         int res=0;
        vector<vector<int>> dp(n,vector<int>(n,0));
        //dp[i][j]:i-j是否是字符串
        for(int i=0;i<n;i++)
        {
            dp[i][i]=1;
        }
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<i;j++)
            {
                if(s[i]!=s[j])
                    dp[j][i]=0;
                else
                {
                    if(i-j<3)
                        dp[j][i]=1;
                    else    
                        dp[j][i]=dp[j+1][i-1];
                }
                if(dp[j][i]==1)
                    res+=1;
            }
        }
       
        for(int i=0;i<n;i++)
        {
            for(int j=i;j<n;j++)
            {
                if(dp[i][j]==1)
                    res+=1;
            }
        }
        return res;
    }
};
```

### 128、最长连续序列（哈希）

```c++
/*
哈希表查找的时间复杂度为O(1)，因此考虑使用哈希表查找连续的数字。

1、将数组数字插入到哈希表，
2、每次随便拿出一个，删除其连续的数字，直至找不到连续的，
3、记录删除的长度，可以找到最长连续序列。
*/
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> q;
        for (int i = 0; i < nums.size(); i++) {
            q.insert(nums[i]);
        }
        int ans = 0;
        while (!q.empty()) {
            int now = *q.begin();
            q.erase(now);
            int l = now - 1, r = now + 1;
            while (q.find(l) != q.end()) {
                q.erase(l);
                l--;
            }
            while(q.find(r) != q.end()) {
                q.erase(r);
                r++;
            }
            l = l + 1, r = r - 1;
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```



### 141、环形链表

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow=head;
        ListNode *fast=head;
        while(fast!=NULL&&fast->next!=NULL)
        {
            slow=slow->next;
            fast=fast->next->next;
            if(slow==fast)
                return true;
        }
        return false;
    }
};
```

### 234、回文链表（快慢指针、反转）

```c++
//先把链表转换成数组，再比较数组里的数是否是回文数组
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> vec;
        ListNode *p=head;
        while(p!=NULL)
        {
            vec.push_back(p->val);
            p=p->next;
        }
        int n=vec.size();
        for (int i = 0, j = n - 1; i < j; i++, j--) 
        {
            if (vec[i] != vec[j]) 
                return false;
        }
        return true;
    
    }
};

//解法2
//利用快慢指针，反转
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if (head == nullptr) {
            return true;
        }

        // 找到前半部分链表的尾节点并反转后半部分链表
        ListNode* firstHalfEnd = endOfFirstHalf(head);
        ListNode* secondHalfStart = reverseList(firstHalfEnd->next);

        // 判断是否回文
        ListNode* p1 = head;
        ListNode* p2 = secondHalfStart;
        bool result = true;
        while (result && p2 != nullptr) {
            if (p1->val != p2->val) {
                result = false;
            }
            p1 = p1->next;
            p2 = p2->next;
        }        

        // 还原链表并返回结果
        firstHalfEnd->next = reverseList(secondHalfStart);
        return result;
    }

    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        while (curr != nullptr) {
            ListNode* nextTemp = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    ListNode* endOfFirstHalf(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
        }
        return slow;
    }
};

```

### 1423、可获得的最大点数

```
//找到n-k长度的最小值
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int sum = 0;
        for (int p : cardPoints)
        {
            sum += p;
        }
        int n = cardPoints.size();

        // 滑动窗口的大小
        int ws = n - k;
        int minSum = 0;
        for (int i = 0; i < ws; ++i)
        {
            minSum += cardPoints[i];
        }
        int currSum = minSum;
        for (int i = ws; i < n; ++i)
        {
            currSum += cardPoints[i] - cardPoints[i-ws];
            minSum = min(minSum, currSum);
        }
        return sum - minSum;

    }
};
```

### 48、旋转图像

```c++
//方法一：先倒置矩阵，然后以中心轴进行对折
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int rows=matrix.size();
        int cols=matrix[0].size();
        //倒置
        for(int row=0;row<rows;row++)
        {
            for(int col=row+1;col<cols;col++)
            {
                swap(matrix[row][col],matrix[col][row]);
            }
        }
        //以中心轴进行对折
        for(int row=0;row<rows;row++)
        {
            for(int col=0;col<cols/2;col++)
            {
                swap(matrix[row][col],matrix[row][cols-col-1]);
            }
        }
    }
};

```

### 238、除自身以外数组的乘积（左右积合并）

```c++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        int n=a.size();
        vector<int> dp1(n,1); //代表下标为i时的左积是多少
        vector<int> dp2(n,1); //代表下标为I时的右积是多少

        //i=0时没有左积
        //i=n-1时没有右积
        
        for(int i=1;i<n;i++)
        {
            dp1[i]=dp1[i-1]*a[i-1];
        }
        for(int i=n-2;i>=0;i--)
        {
            dp2[i]=dp2[i+1]*a[i+1];
        }
        for(int i=0;i<n;i++)
        {
            dp1[i]*=dp2[i];
        }
        return dp1;
    }
};
```

### 64、最小路径和（DP）

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        //dp[i][j]:到(i,j)位置的最小路径和
        int m=grid.size();
        int n=grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        dp[0][0] = grid[0][0];
        for(int i=1;i<n;i++)
        {
            dp[0][i]=dp[0][i-1]+grid[0][i];
        }
        for(int i=1;i<m;i++)
        {
            dp[i][0]=dp[i-1][0]+grid[i][0];
        }
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                dp[i][j]=min(dp[i-1][j]+grid[i][j],dp[i][j-1]+grid[i][j]);
            }
        }
        return dp[m-1][n-1];
    }
};
```

### 14、最长公共前缀

```c++
class Solution {
public:
	//先求两个字符串的最长公共前缀
    string longestCommonPrefix(const string& s1,const string& s2)
    {
        int length=min(s1.size(),s2.size());
        int index=0;
        while(index<length&&s1[index]==s2[index])
        {
            index++;
        }
        return s1.substr(0,index);
    }
    string longestCommonPrefix(vector<string>& strs) {
        if(strs.size()==0)
        {
            return "";
        }
        string prefix=strs[0];
        int count=strs.size();
        int index=0;   //最长前缀索引
        for(int i=1;i<count;i++)
        {
            prefix=longestCommonPrefix(prefix,strs[i]);
            if(prefix.size()==0)
            {
                break;
            }
        }
        return prefix;
    }
};
```

### 13、罗马数字转整数

```
class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char,int> m = {{'I',1}, {'V',5}, {'X',10},{'L',50},{'C',100}, {'D',500},{'M',1000}};
        int sum=0;
        int size=s.size();
        for(int i=0;i<size;)
        {
            if((i<size-1)&&m[s[i]]<m[s[i+1]])
            {
                sum+=m[s[i+1]]-m[s[i]];   //加两步
                i+=2;
            }
            else
            {
                sum += m[s[i]];   //加一步
                i++;
            }
        }
        return sum;
    }
};
```

### 88 、合并排序的数组（双指针）

```c++
//法一
class Solution {
public:
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        for(int i=0;i<n;i++)
        {
            A[i+m]=B[i];
        }
        sort(A.begin(),A.end());
    }
};
//法二：双指针
//思路：从后往前合并，因为是有序的，当其中一个合并完的时候另外一个直接合并到nums1
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int p1=m-1;
        int p2=n-1;
        int cur=m+n-1;
        while(p1>=0&&p2>=0)
        {
            if(nums1[p1]>nums2[p2])
            {
                nums1[cur]=nums1[p1];
                cur--;
                p1--;
            }
            else
            {
                nums1[cur]=nums2[p2];
                cur--;
                p2--;
            }
        }
        while(p1>=0) nums1[cur--]=nums1[p1--];
        while(p2>=0) nums1[cur--]=nums2[p2--];
    }
};
```

### 76、最小覆盖字串（双指针 滑动窗口）

```c++
/*
https://leetcode-cn.com/problems/minimum-window-substring/solution/tong-su-qie-xiang-xi-de-miao-shu-hua-dong-chuang-k/

思路：利用滑动窗口
步骤一：利用i j维护一个窗口，当窗口中增加一个元素就--(意味着我少需要一个)，减少一个元素就++，初始化count为t的size
*/

class Solution {
public:
    string minWindow(string s, string t) {
        //need始终记录着当前滑动窗口下，我们还需要的元素数量
        //结论就是当need中所有元素的数量都小于等于0时，表示当前滑动窗口不再需要任何元素。
        vector<int> need(128,0);
        for(auto c:t)
        {
            need[c]++;
        }
        int count=t.size();  //还需要找多少个
        int l=0,r=0,start=0,size=INT_MAX;
        while(r<s.length())
        {
            char c=s[r];
            if(need[c]>0)  //代表是t里边的元素
                count--;
            need[c]--;  //包含元素之后就要-1
            if(count==0) //证明已经找到一个区间了
            {
                //这个while语句是保证缩小区间，让区间的起始值是有效值
                while(l<r&&need[s[l]]<0)
                {                   
                    need[s[l]]++;  //左指针右移就要++
                    l++;
                }
                //移完之后该更新size
                if(r-l+1<size)
                {
                    size=r-l+1;
                    start=l;
                }
                //再向前走
                need[s[l]]++;
                l++;
                count++;  //因为把最边界的有效元素移出去了
            }
            r++;

        }
        return size==INT_MAX ? "" : s.substr(start, size);
    }
};
```



### 66、加一

```
class Solution {
public:
//分为三种情况
    vector<int> plusOne(vector<int>& digits) {
        int length=digits.size();
        for(int i=length-1;i>=0;i--)
        {
            digits[i]++;
            digits[i]%=10;  //49
            if(digits[i]!=0)  //eg:56
            {
                return digits;
            }
        }
        digits = vector<int>(length + 1);  //99
        digits[0]=1;
        return digits;
    }
};
```

### 11、盛最多水的容器（双指针）

```c++
//解题思路：刚开始指针指向左右两端，每次移动小值
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left=0;
        int right=height.size()-1;
        int ans=0;

        while(left<right)
        {
            int area=min(height[left],height[right])*(right-left);
            ans=max(ans,area);
            if(height[left]<=height[right])
            {
                left++;
            }
            else
            {
                right--;
            }
        }
        return ans;
    }
};
```

### 剑指offer 03、数组中重复的数字（哈希表）

```c++
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        unordered_map<int,int> mp;
        int n=nums.size();
        for(int i=0;i<n;i++)
        {
            mp[nums[i]]++;
            if(mp[nums[i]]>1)
            {
                return nums[i];
            }
        }
        return 0;
    }
};

//排序
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        for(int i=1;i<nums.size();i++)
        {   
            if(nums[i-1] == nums[i]) 
                return nums[i];
        }
        return 0;
    }
};
//原地置换
//思路：一个萝卜一个坑，因为数字范围都是0-（n-1），如果这个坑里边有数字了就说明出现重复的了
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        int size=nums.size();
        for(int i=0;i<size;i++)
        {
            while(nums[i]!=i)  //一直等当前位置下标等于当前数字
            {
                int tmp=nums[i];
                if(tmp==nums[tmp]) 
                {
                    return tmp;
                }
                else
                {
                    swap(tmp,nums[tmp]);
                }
            }
        }
        return 0;
    }
};
```

### 703、数据流中的第K大元素（最小堆）

less默认最大堆，而greater是最小堆。

```
class KthLargest {
public:
    //定义最小堆
    priority_queue<int,vector<int>,greater<int>> q;
    int k;
    KthLargest(int k, vector<int>& nums) {
        this->k=k;
        for(auto &i:nums)
        {
            add(i);
        }
    }
    
    int add(int val) {
        q.push(val);
        if(q.size()>k)
        {
            q.pop();
        }
        return q.top();
    }
};
```

### 767、重构字符串（最大堆）

```c++
class Solution {
public:
    string reorganizeString(string S) {
        string res = "";
        vector<int> counter(26,0);
        priority_queue<pair<int,char>> pq;

        // 统计字母出现的次数
        for(auto& c: S)
            counter[c-'a']++;
        
        // 遍历哈希表
        for(int i = 0;i < 26;i++){
            // 边界条件
            if(counter[i] > (S.size()+1)/2)
                return "";

            // 把字母添加到优先队列中
            if(counter[i] > 0) 
                pq.push({counter[i],i+'a'});
        }
        
        pair<int,char> prev(0,' ');
        
        // 开始重构字符串
        while(!pq.empty()){
            pair<int,char> cur = pq.top();
            pq.pop();
            res += cur.second;
			//保证要先弹出第二大再压入-1操作这个，这样保证不相同
            cur.first--;
            if(prev.first > 0)
                pq.push(prev);
            prev = cur;
        }
        return res;
    }
};

 
```

### 34、在排序数组中查找数字出现的第一个位置和最后一个位置（二分）

```c++
//大雪菜的方法
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) return {-1,-1};
        int left=0,right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]>=target) right=mid;
            else left=mid+1;
        }
        if(nums[right]!=target) return {-1,-1};
        int start=right;

        left=0;
        right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right+1)/2;
            if(nums[mid]<=target) left=mid;
            else right=mid-1;
        }
        int end=right;
        return {start,end};
    }
};


class Solution {
public:
    vector<int> searchRange(vector<int> &nums, int target) {
        if (nums.empty()) {
            return vector<int>{-1, -1};
        }

        int firstPosition = findFirstPosition(nums, target);
        // 如果第 1 次出现的位置都找不到，肯定不存在最后 1 次出现的位置
        if (firstPosition == -1) {
            return vector<int>{-1, -1};
        }
        int lastPosition = findLastPosition(nums, target);
        return vector<int>{firstPosition, lastPosition};
    }

private:
    int findFirstPosition(vector<int> &nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                // ① 不可以直接返回，应该继续向左边找，即 [left, mid - 1] 区间里找
                right = mid - 1;
            } else if (nums[mid] < target) {
                // 应该继续向右边找，即 [mid + 1, right] 区间里找
                left = mid + 1;
            } else {
                // 此时 nums[mid] > target，应该继续向左边找，即 [left, mid - 1] 区间里找
                right = mid - 1;
            }
        }

        // 此时 left 和 right 的位置关系是 [right, left]，注意上面的 ①，此时 left 才是第 1 次元素出现的位置
        // 因此还需要特别做一次判断
        if (left != nums.size() && nums[left] == target) {
            return left;
        }
        return -1;
    }

    int findLastPosition(vector<int> &nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                // 只有这里不一样：不可以直接返回，应该继续向右边找，即 [mid + 1, right] 区间里找
                left = mid + 1;
            } else if (nums[mid] < target) {
                // 应该继续向右边找，即 [mid + 1, right] 区间里找
                left = mid + 1;
            } else {
                // 此时 nums[mid] > target，应该继续向左边找，即 [left, mid - 1] 区间里找
                right = mid - 1;
            }
        }
        // 由于 findFirstPosition 方法可以返回是否找到，这里无需单独再做判断
        return right;
    }
};

```

### 119、杨辉三角II

```
class Solution {
public:
    vector<int> getRow(int rowIndex) {
        vector<int> row(rowIndex + 1);
        row[0] = 1;
        for (int i = 1; i <= rowIndex; ++i)
        {
            for (int j = i; j > 0; --j)    //从右往左
            {
                row[j] += row[j-1];
            }
        }

        return row;
    }
};

```

### 118、杨辉三角（DP）

```
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> res;
        for(int i=0;i<numRows;i++)
        	res.push_back(vector<int>(i+1,1));

       	for(int i=0;i<numRows;i++)
       		for (int j = 1; j < i; j++)
       		 res[i][j]=res[i-1][j-1]+res[i-1][j];
       	return res;
    }
};
```

### 38、外观数列（递归）

```
class Solution {
public:
    string countAndSay(int n) {
        string target;
        if(n==1)
            return "1";
        string before_string=countAndSay(n-1);
        int start=0;
        int end=0;
        char first_char=before_string[0];
        while(end<before_string.size())
        {
            while(first_char==before_string[end++])
            {
                
            }
            end-=1;
            target.push_back('0'+end-start);
            target.push_back(first_char);
            first_char=before_string[end];
            start=end;

        }
        return target;
    }
};
```

### 24、两两交换链表中的节点（递归）

```c++

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        //递归结束条件
        if(head==nullptr||head->next==nullptr)
        {                 
            return head;
        }
        ListNode *newnode=head->next;
        head->next=swapPairs(newnode->next);
        newnode->next=head;
        return newnode;
    }
};
//法二：移动指针
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        auto dummy=new ListNode(-1);
        dummy->next=head;
        for(auto p=dummy;p->next&&p->next->next;)
        {
            auto a=p->next;
            auto b=a->next;
            p->next=b;
            a->next=b->next;
            b->next=a;
            p=a;
        }
        return dummy->next;
    }
};
```

### 58、最后一个单词的长度（双指针）

```
//分为两种情况
“hello world”、"hello world  "、"hello"
class Solution {
public:
    int lengthOfLastWord(string s) {
        int end=s.size()-1;
        int count=0;
        while(end>=0&&s[end]==' ') end--;
        if(end<0) return 0;
        int start=end;
        while(start>=0&&s[start]!=' ')
            start--;
        return end-start;
    }
};
```

### 剑指offer 57、和为S的连续正数序列（双指针）

```c++
//滑动窗口
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        int i=1;
        int j=1;
        int sum=0;
        vector<vector<int>> res;
        while(i<=target/2)
        {
            //右移
            if(sum<target)
            {
                sum+=j;
                j++;
            }
            //左移
            else if(sum>target)
            {
                sum-=i;  //注意顺序不能反
                i++;
            }
            else
            {
                vector<int> path;
                for(int m=i;m<j;m++)
                {
                    path.push_back(m);
                }
                res.push_back(path);
                //右移
                sum-=i; //统计完之后还要右移
                i++;
            }
        }
        return res;
    }
};
```



### 8、字符串转换成整数

```c++
class Solution {
public:
    int myAtoi(string s) {
        int i=0; 
        while(i<s.size()&&s[i]==' ') i++; 
        if(i==s.size()) return 0; 
        
        int sign=1;
        if(s[i]=='-')
        {
            sign=-1;
            i++;
        }
        else if(s[i]=='+')
            i++;
        else if(!isdigit(s[i]))//判断是否为十进制数符(0~9)
            return 0;

        int n=0;
        while(isdigit(s[i])&&i<s.size())
        {
            if((INT_MAX-(s[i]-'0'))/10.0<n) 
            {
            return sign==-1?sign*INT_MAX-1:INT_MAX; 
            }
            n=10*n+(s[i]-'0');
            i++;
        }
        return sign*n;
    }
};
```

### 125、验证回文串（双指针）

```c++
//思路：转换成小写，去除多余的字符生成一个合法的字符串，判断这个字符串是否是回文
class Solution {
public:
    bool isPalindrome(string s) {
        if (s.size() == 0) return true;
        
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
        }

        string str = "";
        for (int i = 0; i < s.size(); i++) {
            if (judge(s[i])) str += s[i];
        }
        
        int i = 0, j = str.size() - 1;
        
        while (i < j) {
            if (str[i++] != str[j--]) return false; 
        }
        return true;
    }
    bool judge(char ch) {
        return ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9'));
    }
};

```

### 100、相同的树

```
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(p==NULL&&q==NULL)
            return true;
        if(p==NULL&&q!=NULL)
            return false;
        if(p!=NULL&&q==NULL)
            return false;
        if(p->val!=q->val)
            return false;
        return isSameTree(p->left,q->left)&&isSameTree(p->right,q->right);


    }
};
```

### 剑指offer 22、链表中倒数第K个节点（双指针）

```c++
//思路：快指针先走K个单位长度，然后快慢指针一块走
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *fast=head;
        ListNode *slow=head;
        while(k--)
        {
            fast=fast->next;
        }
        while(fast)
        {
            fast=fast->next;
            slow=slow->next;
        }
        return slow;
    }
};
```

### 561、数组拆分|

```c++
class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        int n=nums.size();
        int result=0;
        sort(nums.begin(),nums.end());
        for(int i=0;i<n;i++)
        {
            result+=nums[i];
            i+=1;
        }
        return result;
    }
};
```

### 75、颜色分类

```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            while (i <= p2 && nums[i] == 2) {
                swap(nums[i], nums[p2]);
                --p2;
            }
            if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                ++p0;
            }
        }
    }
};
//方法二：单指针+两次遍历
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n=nums.size();
        int pos=0;
        for(int i=0;i<n;i++)
        {
            if(nums[i]==0)
            {
                swap(nums[i],nums[pos]);
                pos++;
            }
        }
        for(int i=pos;i<n;i++)
        {
            if(nums[i]==1)
            {
                swap(nums[i],nums[pos]);
                pos++;
            }
        }
    }
};
```

### 739、每日温度（单调递减栈）

https://leetcode-cn.com/problems/daily-temperatures/solution/leetcode-tu-jie-739mei-ri-wen-du-by-misterbooo/

```c++
//从下到上的元素都是递减的
//我们用栈存储元素的下标，当当前元素>栈顶的元素时----证明找到了-元素下标相减，找到的这个位置是栈顶的下标
//否则就入栈
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res(n, 0);
        stack<int> st;
        for (int i = 0; i < temperatures.size(); ++i) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                auto t = st.top(); 
                st.pop();
                res[t] = i - t;
            }
            st.push(i);
        }
        return res;
    }
};
```

### 69、X的平方根

```c++

//牛顿迭代法
class Solution {
public:
    int mySqrt(int x) {
        if(x==0)
            return 0;
        double C=x,x0=x;
        while(true)
        {
            double xi = 0.5 * (x0 + C / x0);   //求得的方程
            if(abs(xi-x0)<1e-7)
            {
                 break;
            }
            x0=xi;
        }
        return (int)x0;
    }
};
//二分法
class Solution {
public:
    int mySqrt(int x) {
        int l=0,r=x;
        while(l<r)
        {
            int m=(r+(long long)l+1)/2;
            if(m<=x/m) l=m;
            else r=m-1;
        }
        return r;
    }
};
```

### 392、判断子序列（双指针、DP）

```c++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        if(s.size()==0)
            return true;
        if(t.size()==0)
            return false;
        int p1=0;
        int p2=0;
        int count=0;
        while(p1<s.size()&&p2<t.size())
        {
            if(s[p1]==t[p2])
            {
                p1++;
                count++;
            }
            p2++;
        }
        return count==s.size();
    }
};
//动态规划
//dp[i][j]:表示s[0....i]是否是t[0.....j]的子序列
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int lenS = s.size();
        int lenT = t.size();
        if(lenS == 0){
            return true;
        }
        if(lenT == 0){
            return false;
        }
        vector<vector<bool>> dp(lenS + 1, vector<bool>(lenT + 1, false));
        for(int j = 0; j <= lenT; j ++){
            dp[0][j] = true;
        }
        for(int i = 1; i <= lenS; i ++){
            for(int j = 1; j <= lenT; j++){
                if(s[i - 1] == t[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1];
                }else{
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[lenS][lenT];


    }
};
```

### 120、三角形最小路径和（DP）

```c++
//解法一
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n=triangle.size();
        //dp[i]表示：每一行中第i个位置的最小值
        vector<int> dp(n,INT_MAX);
        dp[0]=triangle[0][0];
        for(int i=1;i<n;i++)
        {
            //在最右边的时候
            dp[i]=dp[i-1]+triangle[i][i];
            for(int j=i-1;j>0;j--)
            {
                dp[j]=min(dp[j-1],dp[j])+triangle[i][j];
            }
            dp[0]+=triangle[i][0];
        }
        return *min_element(dp.begin(), dp.end());

    }
};
//解法二：推荐
/*
思路：从下往上计算
(i,j)的下一行相邻的元素是：(i+1,j),(i+1,j+1);
dp[i][j]:从下往上走到(i,j)时的最小路径和，转移方程是：dp[i][j]=(i,j)+min(dp[i+1][j],dp[i+1][j+1])
*/

class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int m=triangle.size();
        for(int i=m-2;i>=0;i--)
        {
            for(int j=0;j<triangle[i].size();j++)
            {
                triangle[i][j]+=min(triangle[i+1][j],triangle[i+1][j+1]);
            }
        }
        return triangle[0][0];
    }
};
```

### 剑指offer 40、最小的K个数(最小堆：顶层为最小元素)

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        priority_queue<int,vector<int>,greater<int>> que;  //小顶堆
        vector<int> res;
        for(auto n:arr)
        {
            que.push(n);
        }
        for(int i=0;i<k;i++)
        {
            res.push_back(que.top());
            que.pop();
        }
        return res;
    }
};
```

### 1046、最后一块石头的重量（最大堆）

```
class Solution {
public:
    int lastStoneWeight(vector<int>& stones) {
        priority_queue<int> que;  //最大堆
        if(stones.size()==2)
            return abs(stones[0]-stones[1]);
        if(stones.size()==1)
            return stones[0];
        for(int i=0;i<stones.size();i++)
        {
            que.push(stones[i]);
        }
        while(que.size()>1)
        {
            int a=que.top();
            que.pop();
            int b=que.top();
            que.pop();
            if(a>b)
            {
                que.push(a-b);
            }

        }
        return que.empty() ? 0 : que.top();
    }
};
```

### 973、最接近原点的K个点（大顶堆）

less-------大顶堆--------栈顶元素最大--------增加数的时候会把大数弹出去，数组里剩下小数

```c++
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        //优先队列 大顶堆
        priority_queue<pair<int,vector<int>>,vector<pair<int,vector<int>>>,less<pair<int,vector<int>>>> que;
        //放元素
        for(int i=0;i<points.size();i++)
        {
            int x=points[i][0];
            int y=points[i][1];
            //先放到一个pair里
            pair<int,vector<int>> p(x*x+y*y,points[i]);
            que.push(p);
            if(que.size()>K)
            {
                que.pop();
            }
        }
        //取元素
        vector<vector<int>> res(K);
        for(int i=0;i<K;i++)
        {
            res[i]=que.top().second;
            que.pop();
        }
        return res;
        
    }
};
```

### 442、数组中重复的数据

```c++
//思路：遍历一遍数组，每次将abs(nums[i])-1位置的数字变成相反数，若发现那个位置上的数字变为正时，哪个位置就是答案
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        int n=nums.size();
        for(int i=0;i<nums.size();i++)
        {
            int pos=abs(nums[i])-1;
            nums[pos]=-nums[pos];
            if(nums[pos]>0)
                res.push_back(pos+1);
        }
        return res;
    }
};
```

### 91、解码方法(DP)

```c++
/*
f[i]  表示前 i 个数字共有多少种解码方式
如果第 i 个数字不是0，则 i 个数字可以单独解码成一个字母，此时的方案数等于用前 i−1 个数字解码的方案数，即 f[i−1]；
如果第 i−1个数字和第 i个数字组成的两位数在 10 到 26之间，则可以将这两位数字解码成一个字符，此时的方案数等于用前 i−2个数字解码的方案数，即 f[i−2]；
*/
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        s = " " + s;
        vector<int> f(n + 1,0);
        f[0] = 1;        
        for(int i = 1; i < n + 1; i++) {
            int a = s[i] - '0', b = (s[i - 1] - '0') * 10 + s[i] - '0';
            if(1 <= a && a <= 9) f[i] = f[i - 1];
            if(10 <= b && b <= 26) f[i] += f[i - 2];
        }
        return f[n];
    }
};
```



### 451、根据字符出现频率排序

```c++
//1、根据出现频率排序，所以我们肯定会想放到hash_map中，分别计算出现次数；
2、根据出现次数从大到小排序，所以我们会用priority_queue比较出现次数排序，次数可以输入pair
3、根据每个pair中的出现次数来确定此字符是否结束；
class Solution {
public:
    string ans;
    string frequencySort(string s) {
        unordered_map<char,int> hash_map;
        for(auto i:s)
        {
            hash_map[i]++;
        }
        //定义优先队列进行排序
        priority_queue<pair<int,char>,vector<pair<int,char>>,less<pair<int,char>>> pq;   //大顶堆

        for(auto tem=hash_map.begin();tem!=hash_map.end();tem++)
        {
            pq.push(make_pair(tem->second,tem->first));
        }
        while(!pq.empty())
        {
            int top_num = pq.top().first;
            char top_char = pq.top().second;
            while(top_num != 0) {
                ans.push_back(top_char);
                top_num -= 1;
            }
            pq.pop();
        }
        return ans;

    }
};
```



### 剑指offer 04、二维数组的查找

```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i=matrix.size()-1;
        int j=0;
        while(i>=0&&j<matrix[0].size())
        {
            if(matrix[i][j]>target) i--;
            else if(matrix[i][j]<target) j++;
            else return true;
        }
        return false;
    }
};
```

### 面试题 0202、返回倒数第K个节点（双指针）

```c++
//先走k步
class Solution {
public:
    int kthToLast(ListNode* head, int k) {
        ListNode* slow=head;
        ListNode* fast=head;
        while(k--)
        {
            fast=fast->next;
        }
        while(fast!=NULL)
        {
            fast=fast->next;
            slow=slow->next;
        }
        return slow->val;
    }
};
```

### 977、有序数组的平方

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        for(int i=0;i<nums.size();i++)
        {
            nums[i]*=nums[i];
        }
        sort(nums.begin(),nums.end());
        return nums;
    }
};

//双指针
//1、i指向起始位置，j指向终止位置。
//2、定义一个新数组result，和A数组一样的大小，让k指向result数组终止位置。
//3、如果A[i] * A[i] < A[j] * A[j] 那么result[k--] = A[j] * A[j]; 。
//4、如果A[i] * A[i] >= A[j] * A[j] 那么result[k--] = A[i] * A[i]; 。
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> res(nums.size());
        int i=0;
        int j=nums.size()-1;
        int k=res.size()-1;
        while(i<=j)
        {
            if(nums[i]*nums[i]>nums[j]*nums[j])
            {
                res[k]=nums[i]*nums[i];
                k--;
                i++;
            }
            else
            {
                res[k]=nums[j]*nums[j];
                k--;
                j--;
            }
        }
        return res;
    }
};
```

### 面试题17.11、单词距离（双指针）

```
class Solution {
public:
    int findClosest(vector<string>& words, string word1, string word2) {
        int ans=INT_MAX;
        int a=-1,b=-1;   //a表示word1的下标，b表示word2的下标
        for(int i=0;i<words.size();i++)
        {
            if(words[i]==word1)
            {
                a=i;
                if(b!=-1)
                    ans=min(ans,abs(a-b));
            }
            if(words[i]==word2)
            {
                b=i;
                if(a!=-1)
                    ans=min(ans,abs(b-a));
            }
        }
        return ans;
    }
};
```

### 面试题08.01、三步问题

```
class Solution {
public:
    int waysToStep(int n) {
        vector<int> dp(n+1,0);
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        if (n == 3) {
            return 4;
        }
        dp[1]=1;
        dp[2]=2;
        dp[3]=4;
        
        for (int i = 4; i < n+1; i++) {
            //取模，对两个较大的数之和取模再对整体取模，防止越界（这里也是有讲究的）
            //假如对三个dp[i-n]都 % 1000000007，那么也是会出现越界情况（导致溢出变为负数的问题）
            //因为如果本来三个dp[i-n]都接近 1000000007 那么取模后仍然不变，但三个相加则溢出
            //但对两个较大的dp[i-n]:dp[i-2],dp[i-3]之和mod 1000000007，那么这两个较大的数相加大于 1000000007但又不溢出
            //取模后变成一个很小的数，与dp[i-1]相加也不溢出
            //所以取模操作也需要仔细分析
            dp[i] = (dp[i-1] + (dp[i-2] + dp[i-3]) % 1000000007) % 1000000007;
        }
        return dp[n];
    }
};
```

### 697、数组的度

· 先求原数组的度；

· 再求与原数组相同***\*度\****的最短子数组。

```c++
//使用 left 和 right 分别保存了每个元素在数组中第一次出现的位置和最后一次出现的位置；使用counter 保存每个元素出现的次数。

//数组的度 degree 等于 counter.values() 的最大值；对counter再次遍历：

//如果元素 k 出现的次数等于 degree，则找出元素 k 最后一次出现的位置 和 第一次出现的位置，计算两者之差+1，即为子数组长度。对所有出现次数等于 degree 的子数组的最短长度，取 min。
class Solution {
public:
    int findShortestSubArray(vector<int>& nums) {
        unordered_map<int, int> left, right, counter;
        int degree = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (!left.count(nums[i]))
                left[nums[i]] = i;
            right[nums[i]] = i;
            counter[nums[i]] ++;
            degree = max(degree, counter[nums[i]]);
        }
        int res = nums.size();
        for (auto& kv : counter) {
            if (kv.second == degree) {
                res = min(res, right[kv.first] - left[kv.first] + 1);
            }
        }
        return res;
    }
};
```

### 387、字符串中的第一个唯一字符（哈希表）

```c++
//哈希表两遍遍历
class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<char,int> map;
        for(int i=0;i<s.size();i++)
        {
            map[s[i]]++;
        }
        int i=0;
        for( i=0;i<s.size();i++)
        {
            if(map[s[i]]==1)
                return i;
        }
        if(i==s.size())
            return -1;
        return 0;
    }
};
```

### 378. 有序矩阵中第 K 小的元素（大顶堆）

```c++
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        //每个元素入堆
        priority_queue<int,vector<int>,less<int>> que;
        for(int i=0;i<matrix.size();i++)
        {
            for(int j=0;j<matrix[0].size();j++)
            {
                que.push(matrix[i][j]);
                if(que.size()>k)
                    que.pop();
            }
        }
        return que.top();
    }
};
```

### 1753、移出石子的最大得分（大顶堆）

从堆里边取出前两个堆顶元素，进行相减

```
class Solution {
public:
    int maximumScore(int a, int b, int c) {
        priority_queue<int,vector<int>,less<int>> que;
        
        que.push(a);
        que.push(b);
        que.push(c);

        int score=0;

        while(que.size()>=2)
        {
            int a=que.top();
            que.pop();
            int b=que.top();
            que.pop();
            a--;
            b--;
            score++;
            if(a) que.push(a);
            if(b) que.push(b);
        }
        return score;
    }
};
```

### 面试题17.14、最小的K个数（大顶堆）

```
class Solution {
public:
    vector<int> smallestK(vector<int>& arr, int k) {
        priority_queue<int,vector<int>,less<int>> que;
        vector<int> res(k);
        if(k>arr.size())
            return res;
        
        //放数
        for(int i=0;i<arr.size();i++)
        {
            que.push(arr[i]);
            if(que.size()>k)
            {
                que.pop();
            }
        }

        //取数
        for(int i=0;i<k;i++)
        {
            res[i]=que.top();
            que.pop();
        }
        return res;
    }
};
```



### 766、托普利茨矩阵

```
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                if(matrix[i][j]!=matrix[i-1][j-1])
                    return false;
            }
        }
        return true;
    }
};
```

### 1720、解码异或后的数组

```
class Solution {
public:
    vector<int> decode(vector<int>& encoded, int first) {
        vector<int> res(encoded.size()+1);
        res[0]=first;
        for(int i=0;i<encoded.size();i++)
        {
            res[i+1]=res[i]^encoded[i];
        }
        return res;
    }
};
```

### 1486、数组异或操作

```
class Solution {
public:
    int xorOperation(int n, int start) {
        vector<int> res(n);
        res[0]=start;
        for(int i=1;i<n;i++)
        {
            res[i]=res[i-1]+2;
        }
        int resu=0;
        for(int i=0;i<n;i++)
        {
            resu^=res[i];
        }
        return resu;
    }
};
```

### 面试题16.01、交换数字

```
class Solution {
public:
    vector<int> swapNumbers(vector<int>& numbers) {
        vector<int> res(2);
        res[0]=numbers[0]^numbers[1]^numbers[0];
        res[1]=numbers[0]^numbers[1]^numbers[1];
        return res;
    }
};
```

### 面试题08.04、幂集

```
class Solution {
public:
    void inner(int n, int level, vector<int>& nums, vector<int>& path, vector<vector<int>>& res) {
      if (n == level) {
        res.push_back(path);
        return;
      }

      path.push_back(nums[level]);
      inner(n, level + 1, nums, path, res);
      path.pop_back();
      inner(n, level + 1, nums, path, res);
    }
    vector<vector<int>> subsets(vector<int>& nums) {
      vector<int> path;
      vector<vector<int>> res;

      inner(nums.size(), 0, nums, path, res);

      return res;
    }
};

```

### 264、丑数||（DP、小顶堆）

```c++
//DP
class Solution {
public:
    int nthUglyNumber(int n) {
        int nums[1700];
        int p2 = 0, p3 = 0, p5 = 0;
        nums[0] = 1, nums[1] = 2, nums[2] = 3, nums[3] = 5;
        for(int i = 1; i < n; i++)
        {
            nums[i] = min(min(2*nums[p2], 3*nums[p3]),5*nums[p5]);
            if(nums[i] == 2*nums[p2]) p2++;
            if(nums[i] == 3*nums[p3]) p3++;
            if(nums[i] == 5*nums[p5]) p5++;
        }
        return nums[n - 1];
    }
};
//小顶堆
class Solution {
public:
    int nthUglyNumber(int n) {
       priority_queue <double,vector<double>,greater<double> > q;
        double answer=1;
        for (int i=1;i<n;++i)
        {
            q.push(answer*2);
            q.push(answer*3);
            q.push(answer*5);
            answer=q.top();
            q.pop();
            while (!q.empty() && answer==q.top())
                q.pop();
        }
        return answer;
    }
};
```

### 面试题17.09、第K个数

```
class Solution {
public:
    int getKthMagicNumber(int k) {
        if (k <= 0) return 0;
        
        vector<long long int> nums(k+1, 1);  // 为防止越界，用 long long保存
        int p3 = 0, p5 = 0, p7 = 0;  // 标记"某个素数"的下标
        for (int i = 1; i < k; ++i)
        {
            nums[i] = min(min(3 * nums[p3], 5 * nums[p5]), 7 * nums[p7]);
            if (nums[i] == 3 * nums[p3]) p3++; // p3++是因为由p3所在的素数求得了最小值，故不会再由p3所在的素数求得另一个最小值，下一个最小值可能是3 * nums[p3+1]。下面p5++, p7++同理。
            if (nums[i] == 5 * nums[p5]) p5++;  // 注意此处是if,而不是else if,因为可能3 *nums[p3] == 5 * nums[p5] 或 7 * nums[p7] == 5 * nums[p5]。下面的同理。
            if (nums[i] == 7 * nums[p7]) p7++;
        }
        return nums[k-1];
    }
};
    
```

### 208、实现Tire树（前缀树）

Trie 是一颗非典型的多叉树模型，多叉好理解，即每个结点的分支数量可能为多个。

sea   sells  she

![image-20210301125618447](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210301125618447.png)



```c++
class Trie {
private:
    bool isEnd;
    Trie* next[26];
public:
    Trie() {
        isEnd = false;
        memset(next, 0, sizeof(next));
    }
/*
  这个操作和构建链表很像。首先从根结点的子结点开始与 word 第一个字符进行匹配，一直匹配到前缀链上没有对应的字符，这时开始不断开辟新的结点，直到插入完 word 的最后一个字符，同时还要将最后一个结点isEnd = true;，表示它是一个单词的末尾
*/
    void insert(string word) {
        Trie* node = this;   //先指向根节点
        for (char c : word) {
            if (node->next[c-'a'] == NULL) {
                node->next[c-'a'] = new Trie();
            }
            node = node->next[c-'a'];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Trie* node = this;
        for (char c : word) {
            node = node->next[c - 'a'];
            if (node == NULL) {
                return false;
            }
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Trie* node = this;
        for (char c : prefix) {
            node = node->next[c-'a'];
            if (node == NULL) {
                return false;
            }
        }
        return true;
    }
};
```

### 148、排序链表（归并排序）

O(nlogn)的时间复杂度，因此要利用二分的思想，同时要用常数级的空间复杂度，因此不能使用递归的方式只能使用迭代的方式

数组进行归并排序时，需要一个额外的数组记录归并结果，因此数组归并的空间复杂度是O(n)+O(logn)，而对于链表的话，直接交换引用即可，不用额外的空间保存，所以只需要O(logn)的递归空间复杂度即可

```c++
/*
*递归：自己调用自己（A调用A）
*迭代：每一次迭代的结果会作为下一次迭代的初始值。（A重复调用B）
*/
//递归+归并
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if(head==NULL||head->next==NULL)
            return head;
        //把链表分成两个链表，从中间分开
        ListNode *slow=head;
        ListNode *fast=head;
        ListNode *pre=NULL;
        while(fast!=NULL&&fast->next!=NULL)
        {
            pre=slow;
            slow=slow->next;
            fast=fast->next->next;
        }
        pre->next=nullptr;
        //把两个链表分别排序

        //合并两个排序好的链表
        return merge(sortList(head),sortList(slow));
    }
    ListNode* merge(ListNode* l1,ListNode* l2)
    {
        if(l1==NULL)
            return l2;
        if(l2==NULL)
            return l1;
        if(l1->val<=l2->val)
        {
            l1->next=merge(l1->next,l2);
            return l1;
        }
        else
        {
            l2->next=merge(l1,l2->next);
            return l2;
        }
    }
};
/*
 * 迭代
 */
class Solution {
public:
    //cut n个节点，返回剩下的链表的首节点
    ListNode *cut(ListNode *head,int n)
    {
        ListNode *p=head;
        while(--n&&p)
        {
            p=p->next;
        }
        if(p==nullptr)
            return NULL;
        ListNode *cur=p->next;
        p->next=NULL;
        return cur;
    }
    //迭代合并链表
    ListNode *merge(ListNode *l1,ListNode *l2)
    {
        ListNode *dummy=new ListNode(0);
        ListNode* p=dummy;
        while(l1&&l2)
        {
            if(l1->val<l2->val)
            {
                p->next=l1;
                l1=l1->next;
            }
            else
            {
                p->next=l2;
                l2=l2->next;
            }
			p=p->next;

        }
        p->next=l1 ? l1 : l2;
        return dummy->next;

    }
    ListNode* sortList(ListNode* head) {
        if(head==NULL||head->next==NULL)
        {
            return head;
        }
        int length=0;
        ListNode* p=head;
        //先求得链表的长度，然后根据长度来cut
        while(p)
        {
            length++;
            p=p->next;
        }
/*
*4 3 8 7  1 6 5
*dummy 3 4 7 8 1 6 5
*dummy 3 4 7 8 1 5 6
*dummy 1 3 4 5 6 7 8
*/
        ListNode *dummy=new ListNode(0);
        dummy->next=head;
        for(int size=1;size<length;size*=2)
        {
            ListNode *cur=dummy->next;
            ListNode *tail=dummy;
            while(cur)
            {
                ListNode *left=cur;
                ListNode *right=cut(left,size);
                cur=cut(right,size);
                tail->next=merge(left,right);
                while(tail->next)
                    tail=tail->next;
            }
        }
        return dummy->next;
    }
};
```

### 236、二叉树的最近公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root||root==p||root==q) return root;
        auto left=lowestCommonAncestor(root->left,p,q);
        auto right=lowestCommonAncestor(root->right,p,q);

        if(!left) return right;
        if(!right) return left;
        return root;
    }
};

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==NULL)
            return NULL;
        if(root==p||root==q)
            return root;
        TreeNode* left=lowestCommonAncestor(root->left,p,q);
        TreeNode* right=lowestCommonAncestor(root->right,p,q);

        if(left!=NULL&&right!=NULL)
            return root;
        if(left==NULL&&right==NULL)
            return NULL;
        return left==NULL?right:left;
        
    }
};
```

### 剑指68、二叉搜索树的最近公共祖先

```c++
//思路：当p q的值都大于根节点的值时，说明都在右子树上，因为lowestCommonAncestor函数的作用就是找最近公共祖先，就调用lowestCommonAncestor传入root的右子树
//当不满足这两个条件时，说明p或q的值一个大于一个小于，就说明这时的根就是最近公共祖先
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root->val<p->val&&root->val<q->val)
            return lowestCommonAncestor(root->right,p,q);
        if(root->val>p->val&&root->val>q->val)
            return lowestCommonAncestor(root->left,p,q);
        return root;
    }
};
```



### 19、删除链表的倒数第n个节点（双指针）

```c++

class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
    //一种常用的技巧是添加一个哑节点（dummy node），它的 \textit{next}next 指针指向链表的头节点。这样一来，我们就不需要对头节点进行特殊的判断了
 
        ListNode* p=new ListNode(0);
        p->next=head;
        ListNode* start=p;
        ListNode* end=p;
        while(n--)
        {
            end=end->next;
        }
        while(end->next!=NULL)
        {
            end=end->next;
            start=start->next;
        }
        start->next=start->next->next;
        return p->next;
    }
};
```

### 33、搜索旋转排序数组（二分法）

```c++
/**
思路：首先根据nums[mid]的值和边界进行比较划分为有序区间和无序区间，先判断值是否在有序区间里，如果在的话缩小范围，在进行二分法查找。
*/
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int len=nums.size();
        if(len==0)
            return -1;
        int left=0;
        int right=len-1;
        while(left<right)
        {
            int mid=left+(right-left+1)/2;
            if(nums[mid]<nums[right])
            {
                if(nums[mid]<=target&&nums[right]>=target)
                {
                    left=mid;
                }
                else{
                    right=mid-1;
                }
            }
            else
            {
                if(nums[left]<=target&&nums[mid-1]>=target)
                {
                    right=mid-1;
                }
                else
                {
                    left=mid;
                }
            }
        }
        if(nums[left]==target)
            return left;
        return -1;
    }
};
//先找出来最小值，就相当于用了两次二分法模板
class Solution {
public:
    int search(vector<int>& nums, int target) {
        //先找出来最小值，把区间分成两段
        if(nums.empty()) return -1;
        int left=0,right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]<=nums.back()) right=mid;
            else left=mid+1;
        }
        if(target>nums.back()) 
        {
            left=0;
            right--;
        } 
        else
        {
            right=nums.size()-1;
        }
        //分开之后再判断
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]>=target) right=mid;
            else left=mid+1;
        }
        if(nums[left]==target) return left;
        else return -1;
    }
};
```

### 153、寻找旋转排序数组的最小值（二分法）

```c++
//大雪菜做法
class Solution {
public:
    int findMin(vector<int>& nums) {
        int left=0;
        int right=nums.size()-1;
        while(left<right)
        {
            int mid=(left+right)/2;
            if(nums[mid]<=nums[right]) right=mid;
            else left=mid+1;

        }
        return nums[right];
    }
};

class Solution {
public:
    int findMin(vector<int>& nums) {
        int left=0;
        int right=nums.size()-1;
        while(left<right)
        {
            int mid=left+(right-left)/2;
            if(nums[mid]<nums[right])
            {
                right=mid;
            }
            else
            {
                left=mid+1;
            }
        }
        return nums[left];
    }
};
```

### 154、寻找旋转排序数组的最小值（二分法）||

```c++
class Solution {
public:
    int findMin(vector<int>& numbers) {
        int i=0;
        int j=numbers.size()-1;
        while(i<j)
        {
            int m=(i+j)/2;
            if(numbers[m]<numbers[j]) //此条件成立，nu[m]有可能是最小值，所以j=m
            {
                j=m;
            }
            else if(numbers[m]>numbers[j]) //此条件成立时，num[m]绝对不是最小值，让i的值跳过他
            {
                i=m+1;
            }
            else
                j--;
        }
        return numbers[i];
    }
};
```

### 240、搜索二维矩阵||

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m=matrix.size();
        int n=matrix[0].size();
        int i=m-1;
        int j=0;
        while(i>=0&&j<n)
        {
            if(target>matrix[i][j])
            {
                j++;
            }
            else if(target<matrix[i][j])
            {
                i--;
            }
            else return true;
        }
        return false;
    }
};
```

### 74、搜索二维矩阵

```c++
/*
可以想象把整个矩阵按行展开成一个一维数组，一维数组单增，直接二分即可
*/
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if(matrix.empty()||matrix[0].empty()) return false;
        int n=matrix.size();
        int m=matrix[0].size();
        int left=0,right=n*m-1;
        while(left<right)
        {
            int mid=(left+right)/2;
            if(matrix[mid/m][mid%m]>=target) right=mid;
            else left=mid+1; 
        }
        if(matrix[right/m][right%m]==target) return true;
        return false;
    }
};
```



### 528、按权重随机选择（二分+前缀和）

```c++
/*
	  给定数组是【3,1,2】——>
      前缀和数组【0,3,4,6】——>
      在【0,6）范围生成一个随机数r ——>
          如果r=0、1、2则返回index0
          如果r=3则返回index1
          如果r=4、5则返回index2
*/
class Solution {
public:
    vector<int> s;
    Solution(vector<int>& w) {
        s=w;
        for(int i = 1;i < s.size();i ++) s[i] += s[i - 1];
    }
    
    int pickIndex() {
        int x = rand() % s.back() + 1;
        int l = 0, r = s.size() - 1;
        while(l < r)
        {
            int mid = l + r >> 1;
            if(s[mid] >= x) r = mid;
            else l = mid + 1;
        }
        return l;
    }
}; 
```

### 560、和为K的子数组（前缀和+hash）

### ![image-20210812110426897](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210812110426897.png)

```c++
//思想
/*
题意：有几种 i、j 的组合，使得从第 i 到 j 项的子数组和等于 k。
->有几种 i、j 的组合，满足 prefixSum[j] - prefixSum[i - 1] == kprefixSum[j]−prefixSum[i−1]==k
//之所以用到map,和两数之和的思想差不多，便于查找
如果不用map的话还需要一层遍历
key--前缀和
val--个数
*/
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1;
        int count = 0, pre = 0;
        for (auto& x:nums) {
            pre += x;
            if (mp.find(pre - k) != mp.end()) {
                count += mp[pre - k];
            }
            mp[pre]++;
        }
        return count;
    }
};
```



### 50、Pow(x,n)

```c++
//快速幂
//先求出次数的一般幂
x^n  2n=y-----x^y
class Solution {
public:
    //计算x的n次方
    double myPow(double x, int n) {
        long long N=n;
        if(N>=0)
            return quickMul(x,N);
        else
            return 1.0/quickMul(x,-N);
        
    }
    double quickMul(double x,long long N)
    {
        if(N==0)
            return 1.0;
        double y=quickMul(x,N/2);
        return (N%2==0)?y*y:y*y*x;
    }
};
```



### 146、LRU缓存机制

int get(int key) {
    if (key 不存在) {
        return -1;
    } else {        
        将数据 (key, val) 提到开头；
        return val;
    }
}

void put(int key, int val) {
    Node x = new Node(key, val);
    if (key 已存在) {
        把旧的数据删除；
        将新节点 x 插入到开头；
    } else {
        if (cache 已满) {
            删除链表的最后一个数据腾位置；
            删除 map 中映射到该数据的键；
        } 
        将新节点 x 插入到开头；
        map 中新建 key 对新节点 x 的映射；
    }
}

![image-20210723204031197](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210723204031197.png)

```c++
class LRUCache {
public:
    LRUCache(int capacity) : cap(capacity) {
    }

    int get(int key) {
        if (map.find(key) == map.end()) return -1;
        auto key_value = *map[key];  //索引取值
        cache.erase(map[key]);
        cache.push_front(key_value);
        map[key] = cache.begin();
        return key_value.second;
    }

    void put(int key, int value) {
        if (map.find(key) == map.end()) {
            if (cache.size() == cap) {
                map.erase(cache.back().first);
                cache.pop_back();
            }
        }
        else {
            cache.erase(map[key]);
        }
        cache.push_front({key, value});
        map[key] = cache.begin();
    }
private:
    int cap;
    list<pair<int, int>> cache;
    unordered_map<int, list<pair<int, int>>::iterator> map;
};
```

### 460、LFU缓存

```c++
//使用哈希表
//平衡树：插入和移除节点
namespace 
{
    struct CacheNode{
        int key;
        int value;
        int freq;    //频率
        long tick;   //访问的时间
        bool operator<(const CacheNode& rhs) const{
            //频率小的排在前边，最先移除的
            if(freq<rhs.freq) return true;
            if(freq==rhs.freq) return tick<rhs.tick;  //相同频率下最久没有访问过
            return false;
        }
    };
}
class LFUCache {
private:
    long tick_;
    int capacity_;
    unordered_map<int,CacheNode> m_;
    set<CacheNode> cache_;

    void touch(CacheNode& node)
    {
        cache_.erase(node);
        node.freq++;
        node.tick=++tick_;
        cache_.insert(node);
    }
public:
    LFUCache(int capacity):capacity_(capacity),tick_(0) {

    }
    
    int get(int key) {
        auto it=m_.find(key);
        if(it==m_.cend()) return -1;
        int value=it->second.value;
        touch(it->second);
        return value;
    }
    
    void put(int key, int value) {
        if(capacity_==0) return;
        auto it=m_.find(key);
        if(it!=m_.cend())
        {
            it->second.value=value;
            touch(it->second);
            return;
        }
        if(m_.size()==capacity_)
        {
            const CacheNode& node=*cache_.cbegin();
            m_.erase(node.key);
            cache_.erase(node);
        }
        CacheNode node{key,value,1,++tick_};
        m_[node.key]=node;
        cache_.insert(node);
    }
};

```



### 1143、最长公共子序列

```c++
class Solution {
public:
    //动态规划
    //dp[i][j]:(0..i-1)和(0...j-1)的最长公共子序列为dp[i][j]
    int longestCommonSubsequence(string text1, string text2) {
        int m=text1.size();
        int n=text2.size();
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        
        //初始化
        for(int i=1;i<=m;i++)
        {
            for(int j=1;j<=n;j++)
            {
                if(text1[i-1]==text2[j-1])
                {
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else
                {
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[m][n];


    }
};
```

### 200、岛屿数量

```c++
class Solution {
public:
    bool isArea(vector<vector<char>>& grid,int r,int c)
    {
        return (r>=0&&r<grid.size())&&(c>=0&&c<grid[0].size());
    }
    void dfs(vector<vector<char>>& grid,int r,int c)
    {
        //base case
        if(!isArea(grid,r,c))
            return;
        if(grid[r][c]!='1') 
            return;
        grid[r][c]=2;
        dfs(grid,r+1,c);
        dfs(grid,r-1,c);
        dfs(grid,r,c+1);
        dfs(grid,r,c-1);
    }
    int numIslands(vector<vector<char>>& grid) {
        int count=0;
        int m=grid.size();
        int n=grid[0].size();
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(grid[i][j]=='1')
                {
                    count++;
                    dfs(grid,i,j);
                }
            }
        }
        return count;
    }
};
```

### 329、矩阵中的最长递增路径（记忆化搜索+dp）

```c++
//当我们计算(x,y)的最长路径时，我们必须得知道它的上下左右的结果值，如果它的上下左右的结果值没有的话，就递归求解出-------记忆化搜索
class Solution {
public:
    int n,m;
    vector<vector<int>> dp;
    vector<vector<int>> g;
    int dx[4]={-1,0,1,0};
    int dy[4]={0,1,0,-1};
    int dpfunc(int x,int y)
    {
        if(dp[x][y]!=-1) return dp[x][y];  //当前位置计算过，直接返回
        dp[x][y]=1;
        for(int i=0;i<4;i++)
        {
            int a=x+dx[i];
            int b=y+dy[i];
            if(a>=0&&a<n&&b>=0&&b<m&&g[a][b]<g[x][y])
            {
                dp[x][y]=max(dp[x][y],dpfunc(a,b)+1);
            }
        }
        return dp[x][y];
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        if(matrix.empty()) return 0;
        g=matrix;
        n=g.size();
        m=g[0].size();
        dp=vector<vector<int>>(n,vector<int>(m,-1));
        int res=0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {
                res=max(res,dpfunc(i,j));
            }
        }
        return res;
    }
};
```

### 440、字典序的第K小数字

```c++
class Solution {
public:
    #define LL long long
    int calc(int prefix, int n) {
        int tot = 0;
		//k代表相同相同位数下可以加的数字，比如3位数字，k=100
        //tot当前前缀有多少数字
        //t代表当前前缀每个量级最开始的数1 10 100 1000
        LL t = prefix, k = 1;
        while (t * 10 <= n) {
            tot += k;
            k *= 10;
            t *= 10;
        }

        if (t <= n) { // 此时 t 一定和 n 数字的位数相同
            if (n - t < k)
                tot += n - t + 1;
            else
                tot += k;
        }

        return tot;
    }
    int findKthNumber(int n, int k) {
        int prefix=1;
        while(k>1)
        {
            int sz=calc(prefix,n);
            if(k>sz)
            {
                k-=sz;
                prefix++;
            }
            else{
                k--;
                prefix=prefix*10;
            }
        }
        return prefix;
    }
};
```



### 剑指offer 12、矩阵中的路径

```c++
/*
为何还原元素：
 因为只代表此次搜索过程中，该元素已访问过，当初始i j变化时，又开始了另一次搜索过程
 
 递归搜索匹配字符串过程中，需要 board[i][j] = '/' 来防止 ”走回头路“ 。当匹配字符串不成功时，会回溯返回，   此时需要board[i][j] = tmp 来”取消对此单元格的标记”。 在DFS过程中，每个单元格会多次被访问的，    board[i][j] = '/'只是要保证在当前匹配方案中不要走回头路。

*/
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
private:
    int rows, cols;
    bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
        if(i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';
        bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || 
                      dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];   //还原元素
        return res;
    }
};
 
```

### 剑指offer 13、机器人的运动范围

```c++
class Solution {
private:
    int change(int n)
    {
        int sum = 0;
        while(n > 0) {
            sum += n % 10;
            n /= 10; 
        }
        return sum;
    }
    int dfs(vector<vector<bool>> &visited,int m,int n,int i,int j,int k)
    {
        //注意：i==m时条件也不成立
        if(i>=m||j>=n||visited[i][j]||change(i)+change(j)>k)
            return 0;
        visited[i][j]=true;
        return 1+dfs(visited,m,n,i+1,j,k)+dfs(visited,m,n,i,j+1,k);
    }
public:
    //35--->3+5

    int movingCount(int m, int n, int k) {
        vector<vector<bool>> visited(m, vector<bool>(n, 0));
        return dfs(visited,m,n,0,0,k);
    }
};
```



### 695、岛屿的最大面积

```c++
//类似于树的遍历
class Solution {
public:
    bool isarea(vector<vector<int>>& grid,int r,int c)
    {
        return (r>=0&&r<grid.size())&&(c>=0&&c<grid[0].size());
    }
    int area(vector<vector<int>>& grid,int r,int c)
    {
        if(!isarea(grid,r,c))
            return 0;
        if(grid[r][c]!=1) 
            return 0;
        grid[r][c]=2;
        return 1+area(grid,r+1,c)+area(grid,r-1,c)+area(grid,r,c+1)+area(grid,r,c-1);
        
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int count=0;
        int m=grid.size();
        int n=grid[0].size();
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(grid[i][j]==1)
                {
                    int s=area(grid,i,j);
                    count=max(count,s);
                }
            }
        }
        return count;
    }
};
```

### 463、岛屿的周长

```
class Solution {
public:
    bool isarea(vector<vector<int>>& grid,int r,int c)
    {
        return (r>=0&&r<grid.size())&&(c>=0&&c<grid[0].size());
    }
    int area(vector<vector<int>>& grid,int r,int c)
    {
        if(!isarea(grid,r,c))
            return 1;
        if(grid[r][c]==0) 
            return 1;
        if(grid[r][c]!=1)
            return 0;
        grid[r][c]=2;
        return area(grid,r+1,c)+area(grid,r-1,c)+area(grid,r,c+1)+area(grid,r,c-1);
        
    }
    int islandPerimeter(vector<vector<int>>& grid) {
        int count=0;
        int m=grid.size();
        int n=grid[0].size();
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(grid[i][j]==1)
                {
                    count=area(grid,i,j);
                }
            }
        }
        return count;
    }
};
```

### 剑指 62、圆圈中最后剩下的数字（约瑟夫环问题）[DP]

```c++
//dp[i]=(dp[i-1]+m)%i;
/*
n=1,f(n,m)=0
n>1,f(n,m)=[f(n,m)+m]%n
*/
class Solution {
public:
    int lastRemaining(int n, int m) {
        int res=0;  //被删的数
        for(int i=2;i<=n;i++)
        {
            res=(res+m)%i;
        }
        return res;

    }
};
```

### 1796、字符串中的第二大数字

```
class Solution {
public:
    int secondHighest(string s) {
        set<int> s1;
        for(int i=0;i<s.size();i++)
        {
            if(isdigit(s[i]))
            {
                s1.insert(s[i]-'0');
            }
        }
        if(s1.size()<2) return -1;
        int m=0;
        int res;
        for(auto i=s1.rbegin();i!=s1.rend();i++)
        {
            m++;
            if(m==2)
            {
                 res=*i;
                break;
            }
        }
        return res;
        
        
        
    }
};
```

### 470、用rand7生成rand10

```c++
/*
定理：若rand_n()能等概率生成1到n的随机整数，则有(rand_n() - 1) * n + rand_n()能等概率生成1到n * n的随机整数。
* * * rand()7能等概率生成1~7,
* * * rand7() - 1能等概率生成0~6,
* * * (rand7() - 1) * 7能等概率生成{0, 7, 14, 21, 28, 35, 42},
* * * (rand7() - 1) * 7 + rand7()能等概率生成1~49。
*/
class Solution {
public:
    int rand10() {
        int curr=(rand7()-1)*7+rand7();  //会产生0-49之间的数
        //如果是curr>10的话，效率会很低
        while(curr>40)
        {
            //舍弃41～49，因为是独立事件，我们生成的1～40之间的数它是等概率的
            curr = (rand7() -1 )*7 + rand7();
        }
        return 1 + curr % 10;
    }
};
```



# 算法思想

## 动态规划

- 确定dp数组（dp table）以及下标的含义
- 确定递推公式
- dp数组如何初始化
- 确定遍历顺序
- 举例推导dp数组

### 509、斐波那契数列

```c++
//解法1
class Solution {
public:
    int fib(int n) {
        if(n<=1) return n;
        vector<int> dp(n+1);
        dp[0]=0;
        dp[1]=1;
        for(int i=2;i<=n;i++)
        {
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
};
//解法2
class Solution {
public:
    int fib(int n) {
        if(n==0)
            return 0;
        if (n == 2 || n == 1) 
            return 1;
        int prev = 1, curr = 1;
    	for (int i = 3; i <= n; i++) 
    	{
        	int sum = prev + curr;
        	prev = curr;
        	curr = sum;
   		}
    	return curr;
    }
};
//溢出问题
class Solution {
public:
    int fib(int n) {
        if(n<=1) return n;
        int pre=0;
        int cur=1;
        for(int i=2;i<=n;i++)
        {
            int tmp=(pre+cur)%1000000007;
            pre=cur;
            cur=tmp;
        }
        return cur;
    }
};
```

### 70、爬楼梯(DP)

```c++
class Solution {
public:
    int climbStairs(int n) {
        if(n<=1) return n;
        int dp[3];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            int sum = dp[1] + dp[2];
            dp[1] = dp[2];
            dp[2] = sum;
        }
        return dp[2];
    }
};
```

### 746、使用最小花费爬楼梯

```
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int dp0=cost[0];
        int dp1=cost[1];
        for(int i=2;i<cost.size();i++)
        {
            int dpi=min(dp0,dp1)+cost[i];
            dp0=dp1;
            dp1=dpi;
        }
        return min(dp0,dp1);
    }
};
```

### 62、不同路径

```c++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m,vector<int>(n,0));
        for(int i=0;i<m;i++) dp[i][0]=1;
        for(int j=0;j<n;j++) dp[0][j]=1;
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
//优化后
class Solution {
public:
    //dp[i][j]:到(i,j)共有dp[i][j]条路径
    int uniquePaths(int m, int n) {
        //vector<vector<int>> dp(m,vector<int>(n,0));
        vector<int> dp(n,1);
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                dp[j]+=dp[j-1];
            }
        }
        return dp[n-1];
    }
};
```

### 63、不同路径II (DP)

```c++
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m=obstacleGrid.size();
        int n=obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        for(int i=0;i<m&&obstacleGrid[i][0]!=1;i++) dp[i][0]=1;
        for(int j=0;j<n&&obstacleGrid[0][j]!=1;j++) dp[0][j]=1;

        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                if(obstacleGrid[i][j]==1) continue;
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
//另一种写法
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m=obstacleGrid.size();
        int n=obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,0));
        for(int i=0;i<m;i++)
        {
            if(obstacleGrid[i][0]==0)
                dp[i][0]=1;
            else
                break;
        }
        for(int i=0;i<n;i++)
        {
            if(obstacleGrid[0][i]==0)
                dp[0][i]=1;
            else
                break;
        }
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                if(obstacleGrid[i][j]==0)
                {
                    dp[i][j]=dp[i-1][j]+dp[i][j-1];
                }
                else
                    continue;
            }
        }
        return dp[m-1][n-1];
    }
};
```

### 343、整数拆分

```c++
//DP
class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n+1);
        dp[2]=1;
        for(int i=3;i<=n;i++)
        {
            for(int j=1;j<i-1;j++)
            {
                dp[i]=max(dp[i],max((i-j)*j,dp[i-j]*j));
            }
        }
        return dp[n];
    }
};
//贪心
class Solution {
public:
    int cuttingRope(int n) {
        if(n<=3) return n-1;
        int shang=n/3;
        int yu=n%3;
        if(yu==0) return pow(3, shang);
        else if(yu==1) return pow(3,shang-1)*4;
        else 
        return pow(3,shang)*2;

    }
};
```

### 剑指offer14、剪绳子||（贪心）

```c++
//不能用动态规划，解决不了数溢出的问题
class Solution {
public:
    int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        if(n == 4) return 4;
        long res = 1;
        while(n > 4) 
        {
            res *= 3;
            res %= 1000000007;
            n -= 3;
        }
        // 最后n的值只有可能是：2、3、4。而2、3、4能得到的最大乘积恰恰就是自身值
        // 因为2、3不需要再剪了（剪了反而变小）；4剪成2x2是最大的，2x2恰巧等于4
        return res * n % 1000000007; 
    }
};

```



### 96、不同的二叉搜索树

```
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n + 1);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }
};
```

### 416、分割等和子集

```
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
    for (int num : nums) sum += num;
    // 和为奇数时，不可能划分成两个和相等的集合
    if (sum % 2 != 0) return false;
    int n = nums.size();
    sum = sum / 2;
    vector<vector<bool>> 
        dp(n + 1, vector<bool>(sum + 1, false));
    // base case
    for (int i = 0; i <= n; i++)
        dp[i][0] = true;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
        //因为是前多少数，所以是nums[i-1]
            if (j - nums[i - 1] < 0) {
               // 背包容量不足，不能装入第 i 个物品
                dp[i][j] = dp[i - 1][j]; 
            } else {
                // 装入或不装入背包
                dp[i][j] = dp[i - 1][j] | dp[i - 1][j-nums[i-1]];
            }
        }
    }
    return dp[n][sum];
        
    }
};

class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int num : nums) sum += num;
        // 和为奇数时，不可能划分成两个和相等的集合
        if (sum % 2 != 0) return false;
        int n = nums.size();
        sum = sum / 2;
        //dp[i][j]:前i个数中是否能凑成和为sum
        vector<vector<bool>> 
            dp(n , vector<bool>(sum+1 , false));
        for(int i=0;i<n;i++)
        {
            dp[i][0]=true;
        }
        for(int i=1;i<n;i++)
        {
            for(int j=1;j<=sum;j++)
            {
                
                if(j-nums[i]<0)
                {
                    dp[i][j]=dp[i-1][j];
                }
                else
                {
                    dp[i][j]=dp[i-1][j]|dp[i-1][j-nums[i]];
                }
            }
        }
         return dp[n-1][sum];
    }
};
```



### 1049、最后一块石头的重量II

这个问题可以转换成和416问题一样，划分成两个区间，求区间和的最小值

```
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        vector<int> dp(15001,0);
        int sum=0;
         for (int i = 0; i < stones.size(); i++) sum += stones[i];
        int target=sum/2;
        for(int i=0;i<stones.size();i++)
        {
            for(int j=target;j>=stones[i];j--)
            {
                dp[j]=max(dp[j],dp[j-stones[i]]+stones[i]);
            }
        }
        return sum-dp[target]-dp[target];
    }
};
```



### 322、零钱兑换

```c++
//利用数组来存节点值
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int Max=amount+1;
        int n=coins.size();
        vector<int> dp(amount+1,Max);
         dp[0]=0;
        for(int i=1;i<=amount;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(i>=coins[j]) //证明可以兑换
                {
                    dp[i]=min(dp[i],dp[i-coins[j]]+1);
                }
            }
        }
        if(dp[amount]==Max)
            return -1;
        else
            return dp[amount];
    }
};
```

### 518、零钱兑换||

```c++
//相当于跑楼梯
//分清排列和组合是不同的。
//注意：先列举coin,然后是amount
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1,0);
        dp[0]=1;
        for(auto coin:coins)
        {
            for(int i=1;i<=amount;i++)
            {
                if(i>=coin)
                {
                    dp[i]+=dp[i-coin];
                }
            }
        }
        return dp[amount];
    }
};
 

class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n=coins.size();
        vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
        for (int i = 0; i <= n; i++) 
            dp[i][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++)
            if (j - coins[i-1] >= 0)
                dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i-1]];
            else 
                dp[i][j] = dp[i - 1][j];
        }
        return dp[n][amount];
    }
};
```



### 300、最长递增子序列

```c++
//思路：到第i个数的最长递增子序列=max(前面比i小的数的最长递增子序列)+1
class Solution {
public:

    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(),1);
        //dp[i]代表的意义：nums[i]这个数结尾的最长递增长度
        if(nums.size()==0)
            return 0;
        for(int i=1;i<nums.size();i++)
        {
            for(int j=0;j<i;j++)
            {
                if(nums[j]<nums[i])
                {
                    dp[i]=max(dp[i],1+dp[j]);
                }
            }
        }
        int res=0;
        for(int i=0;i<dp.size();i++)
        {
            res=max(res,dp[i]);
        }
        return res;
    }
};
```

### 718、最长重复子数组

```c++
//dp[i][j]：A的前i个元素和B的前j个之间的最长重复子数组
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        int m=A.size();
        int n=B.size();
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        int max_val=0;
         
         
        for(int i=1;i<=m;i++)
        {
            for(int j=1;j<=n;j++)
            {
                if(A[i-1]==B[j-1])
                {
                    dp[i][j]=1+dp[i-1][j-1];
                }
                else{
                    dp[i][j]=0; //因为要求连续子数组
                }
                max_val=max(max_val,dp[i][j]);
            }
        }
        return max_val;
    }
};
```



### 198、打家劫舍

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
        //dp[i]代表前i间房子能偷到的最大金额
        int size=nums.size();
        vector<int> dp = vector<int>(size, 0);
        
        if(nums.size()==0)
            return 0;
        if(nums.size()==1)
            return nums[0];
        if(nums.size()==2)
            return max(nums[0],nums[1]);

        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        for(int i=2;i<nums.size();i++)
        {
            dp[i]=max(nums[i]+dp[i-2],dp[i-1]);  //选和不选的问题
        }
        return dp[nums.size()-1];
    }
};

//优化后
class Solution {
public:
    int rob(vector<int>& nums) {
        //dp[i]代表前i间房子能偷到的最大金额
        int size=nums.size();
        vector<int> dp = vector<int>(size, 0);
        
        if(nums.size()==0)
            return 0;
        if(nums.size()==1)
            return nums[0];
        if(nums.size()==2)
            return max(nums[0],nums[1]);

        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        int dp_i_1 = 0, dp_i_2 = 0;
        int dp_i = 0;
        for(int i=0;i<nums.size();i++)
        {
            dp_i=max(nums[i]+dp_i_2,dp_i_1);
            dp_i_2=dp_i_1;
            dp_i_1=dp_i;
        }
        return dp_i;
    }
};
```

### 213、打家劫舍II

```c++
class Solution {
public:
    int rob(vector<int>& nums) {
//对于这种情况分两种：选第一个不选最后一个，选最后一个不选第一个
        int length=nums.size();
        if(length==0) return 0;
        if(length==1) return nums[0];
        vector<int> dp(length);
        dp[0]=nums[0];
        dp[1]=max(nums[0],nums[1]);
        for(int i=2;i<length-1;i++)
        {
            dp[i]=max(dp[i-2]+nums[i],dp[i-1]);
        }
        int result=dp[length-2];
        dp[0]=0;dp[1]=nums[1];
        for(int i=2;i<length;i++)
        {
            dp[i]=max(dp[i-2]+nums[i],dp[i-1]);
        }
        result=max(result,dp[length-1]);
        return result;

        
    }
};
```

### 337、打家劫舍III

```c++

class Solution {
public:
//以root为根节点的最大值
    unordered_map<TreeNode*, int> vals;
    int rob(TreeNode* root) {
        if(root==NULL) return 0;
        if(root->left==NULL&&root->right==NULL) return root->val;
        //避免重复计算节点，先判断事先是否存下来了相应节点
        if(vals[root]) return vals[root];

        //偷父节点
        int val1=root->val;
        if(root->left) val1=val1+rob(root->left->left)+rob(root->left->right);
        if(root->right) val1=val1+rob(root->right->left)+rob(root->right->right);

        //不偷父节点
        int val2=0;
        val2=rob(root->left)+rob(root->right);
        vals[root] = max(val1, val2);
        return max(val1,val2);
    }
};
```

### 1025、除数博弈

```
class Solution {
public:
    bool divisorGame(int N) {
        if(N%2==0) return true;
        else return false;
    }
};
```

### 1641、统计字典序元音字符串的数目

```
class Solution {
public:
    int countVowelStrings(int n) {
      vector<vector<int>> dp(n,vector<int>(5));
      dp[0][0]=1;   //以a开始的
      dp[0][1]=1;
      dp[0][2]=1;
      dp[0][3]=1;
      dp[0][4]=1;
      
      for(int i=1;i<n;i++)
      {
          dp[i][0]=dp[i-1][0]+dp[i-1][1]+dp[i-1][2]+dp[i-1][3]+dp[i-1][4];
          dp[i][1]=dp[i-1][1]+dp[i-1][2]+dp[i-1][3]+dp[i-1][4];
          dp[i][2]=dp[i-1][2]+dp[i-1][3]+dp[i-1][4];
          dp[i][3]=dp[i-1][3]+dp[i-1][4];
          dp[i][4]=dp[i-1][4];

      }
      return (dp[n-1][0]+dp[n-1][1]+dp[n-1][2]+dp[n-1][3]+dp[n-1][4]);

    }
};
```

### 338、比特位计数（DP）

```c++
class Solution {
public:
    //统计一个数的二进制位数
    int count_2(int n)
    {
        int count=0;
        while(n)
        {
            n&=(n-1);
            count++;
        }
        return count;
    }
    vector<int> countBits(int num) {
        vector<int> result(num+1);
        result[0]=0;
        for(int i=1;i<=num;i++)
        {
            result[i]=count_2(i);
        }
        return result;

    }
};
//动态规划
class Solution {
public:
    //dp[i]:第i个数的比特位含有1的个数
    vector<int> countBits(int num) {
        vector<int> dp(num+1,0);
        dp[0]=0;
        if(num==0) 
            return dp;
        dp[1]=1;
        if(num==1) 
            return dp;
        dp[2]=1;
        if(num==2) 
            return dp;
        int flag=2;
        for(int i=3;i<=num;i++)
        {
            if(i%flag==0)
                flag*=2;
            dp[i]=1+dp[i-flag];
        }
        return dp;
    }
};
```

### 221、最大正方形

![image-20210807123112571](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210807123112571.png)

```c++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();
        //dp[i][j]:(i,j)坐标正方形的边数
        vector<vector<int>> dp(m+1,vector<int>(n+1,0));
        if(matrix.size()==0)
            return 0;
        //初始化
        int ans=0;  //边的最大值
        for(int i=1;i<=m;i++)
        {
            for(int j=1;j<=n;j++)
            {
                if(matrix[i-1][j-1]=='1')
                {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                    ans=max(ans,dp[i][j]);
                }
            }
        }
        return ans*ans;
    }
};
```

### 剑指offer47、礼物的最大价值

```
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m=grid.size();
        int n=grid[0].size();
        //初始化
        vector<vector<int>> dp(m,vector<int>(n,0));
        dp[0][0]=grid[0][0];
        for(int i=1;i<n;i++)
        {
            dp[0][i]=dp[0][i-1]+grid[0][i];
        }
        for(int i=1;i<m;i++)
        {
            dp[i][0]=dp[i-1][0]+grid[i][0];
        }
        
        for(int i=1;i<m;i++)
        {
            for(int j=1;j<n;j++)
            {
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[m-1][n-1];
    }
};
```

### 剑指offer  64、1+2+3+....n

```c++
class Solution {
public:
    //动态规划
    //dp[i]:到i为止的和
    int sumNums(int n) {
        vector<int> dp(n+1,0);
        dp[0]=0;
        for(int i=1;i<=n;i++)
        {
            dp[i]=dp[i-1]+i;
        }
        return dp[n];
    }
};
//优化
class Solution {
public:
    int sumNums(int n) {
       
        int pre=0;
        for(int i=1;i<=n;i++)
        {
            int cur=pre+i;
            pre=cur;
        }
        return pre;
    }
};
```

### 279、完全平方数个数（DP）

```c++
class Solution {
public:
    int numSquares(int n) {
        //dp[i]:组合成i所对应的平方数个数
        vector<int> dp(n+1,n);
        if(n<=3) return n;
        dp[0]=0;
        dp[1]=1;
        dp[2]=2;
        dp[3]=3;
        for(int i=4;i<=n;i++)
        {
            for(int j=1;j<=sqrt(i);j++)
            {
                if(i>=pow(j,2))
                {
                    dp[i]=min(dp[i],1+dp[i-pow(j,2)]);
                }
            }
        }
        return dp[n];
    }
};
```

### 72、编辑距离（DP）

```c++
/*
2、定义 dp[i][j]
	21. dp[i][j] 代表 word1 中前 i 个字符，变换到 word2 中前 j 个字符，最短需要操作的次数
	22. 需要考虑 word1 或 word2 一个字母都没有，即全增加/删除的情况，所以预留 dp[0][j] 和 dp[i][0]

3、状态转移
	31. 增，dp[i][j] = dp[i][j - 1] + 1
	32. 删，dp[i][j] = dp[i - 1][j] + 1
	33. 改，dp[i][j] = dp[i - 1][j - 1] + 1
	34. 按顺序计算，当计算 dp[i][j] 时，dp[i - 1][j]，dp[i][j - 1]，dp[i - 1][j - 1]均已经确定了
	35. 配合增删改这三种操作，需要对应的 dp 把操作次数加一，取三种的最小
	36.如果刚好这两个字母相同 word1[i-1] = word2[j-1] ，那么可以直接参考 dp[i-1][j-1],操作不用加一
*/
class Solution {
public:
    int minDistance(string word1, string word2) {
       vector<vector<int>> dp(word1.size() + 1, vector<int>(word2.size() + 1, 0));

        for (int i = 0; i < dp.size(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < dp[0].size(); j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i < dp.size(); i++) {
            for (int j = 1; j < dp[i].size(); j++) {
                dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                }
            }
        }
        return dp.back().back();
    }
};
```

### 494、目标和（DP）

```c++
/*
1、dp[i][j]：添加前i个数字的符号，得到总和为j的方案数
2、对于第i个符号，dp[i][j]=dp[i-1][j+nums[i]]+dp[i-1][j-nums[i]]
3、j的范围[-sum,sum],sum为数组元素的总和
4、初始值dp[0][0]=1;最终答案dp[n,s];
5、为了方便，第一维的下标从1开始，第二维也需要给sum的偏移量防止负下标
比如数组的范围是[-100,100],我们需要定义一个偏移量sum=100,将数组范围定义为[0,200]
*/
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n=nums.size();
        int sum=accumulate(nums.begin(),nums.end(),0);
        if(!(target<=sum&&target>=-sum))  //不可能有组合
            return 0;
        vector<vector<int>> dp(n+1,vector<int>(2*sum+1,0));
        dp[0][0+sum]=1;  //证明没有数字添加符号
        for(int i=1;i<=n;i++)
        {
            for(int j=-sum;j<=sum;j++)
            {
                if(j-nums[i-1]>=-sum)
                    dp[i][j+sum]+=dp[i-1][j-nums[i-1]+sum];  //根据下标判断if条件
                if(j+nums[i-1]<=sum)
                    dp[i][j+sum]+=dp[i-1][j+nums[i-1]+sum];
            }
        }
        return dp[n][target+sum];
    }
};
```



## 回溯

### 77、组合问题

https://mp.weixin.qq.com/s/Ri7spcJMUmph4c6XjPWXQA

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5C8W0mj9MSP18fpeFku5B0YHWINvHGHibzoslgQnd9JJNSLO9YuRYmLe93WvPSbrYs4alViczibxPXg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(int n,int k,int startIndex)
    {
        //终止条件
        if(path.size()==k)
        {
            //保存结果
            result.push_back(path);
            return;
        }
        //单层搜索的过程
        for(int i=startIndex;i<=n;i++)
        {
            path.push_back(i);
            backtracking(n,k,i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        backtracking(n, k, 1);
        return result;
    }
};
//剪枝操作
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(int n,int k,int startIndex)
    {
        //终止条件
        if(path.size()==k)
        {
            //保存结果
            result.push_back(path);
            return;
        }
        //单层搜索的过程
        for(int i=startIndex;i<=n-(k-path.size())+1;i++)
        {
            path.push_back(i);
            backtracking(n,k,i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        backtracking(n, k, 1);
        return result;
    }
};
```

### 216、组合求和III

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5r2ImicwOgb3acfYMO7xch11n4Xhtkibhr2VXtdrOQiajKLwXBJglNZ6kGJ5ZL4rqs2T5icX9icPMwnaA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
class Solution {
public:
    vector<vector<int>> result; // 存放结果集 
    vector<int> path; // 符合条件的结果
    void backtracking(int targetSum, int k, int sum, int startIndex) 
    {
        if (sum > targetSum) { // 剪枝操作
            return; // 如果path.size() == k 但sum != targetSum 直接返回
        }
        
        if(path.size()==k)
        {
            if(targetSum==sum) result.push_back(path);
            return;
        }
        for(int i=startIndex;i<=9;i++)
        {
            sum += i;
            path.push_back(i);
            backtracking(targetSum, k, sum, i + 1); // 注意i+1调整startIndex
            sum -= i; // 回溯 
            path.pop_back(); // 回溯 
        }
    }
    vector<vector<int>> combinationSum3(int k, int n) {
        backtracking(n, k, 0, 1);
        return result;
    }
};
//不需要sum
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(int targetsum,int k,int index)
    {
        if(targetsum<0)   //剪枝操作
            return;
       
        if(path.size()==k)
        {
            if(targetsum==0) result.push_back(path);
            return;
        }
        for(int i=index;i<=9;i++)
        {
            targetsum-=i;
            path.push_back(i);
            backtracking(targetsum,k,i+1);
            targetsum+=i;
            path.pop_back();
        }
    }

    //k个集合组成和为n
    vector<vector<int>> combinationSum3(int k, int n) {
       backtracking(n,k,1);
       return result;
    }
};
```

### 17、电话号码的数字组合

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5xxv8pvLj20nF5NlgJF5Je5QF3NVkvfCXWCPB0cqJupql78Xo1cMsD9mH6qibKuicpBohkbVhJcnFQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```c++
class Solution {
private:
    //首先列出数字对应什么字符串
    const string letterMap[10]={
        "", // 0
        "", // 1
        "abc", // 2
        "def", // 3
        "ghi", // 4
        "jkl", // 5
        "mno", // 6
        "pqrs", // 7
        "tuv", // 8
        "wxyz", // 9
    };
public:
    vector<string> result;
    string s;
    //index是遍历到第几个数字
    void backtracking(const string& digits,int index)
    {
        if(index==digits.size())  //也可以是s.size()==digits.size()
        {
            result.push_back(s);
            return;
        }
        int digit = digits[index] - '0';        // 将index指向的数字转为int
        string letters = letterMap[digit];      // 取数字对应的字符集 
        for(int i=0;i<letters.size();i++)
        {
            s.push_back(letters[i]);
            backtracking(digits,index+1);
            s.pop_back();
        }

    }
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) {
            return result;
        }
        backtracking(digits, 0);
        return result;
    }
};
//ysc
class Solution {
public:
    string chars[8]={"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    vector<string> letterCombinations(string digits) {
        if(digits.empty()) return vector<string>();
        vector<string> state(1,"");
        for(auto u:digits)
        {
            vector<string> now;
            for(auto c:chars[u-'2'])
            {
                for(auto s:state)
                {
                    now.push_back(s+c);
                }
            }
            state=now;
        }
        return state;
    }
};
```

### 39、组合总和

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv7XHgvUW5rUaJibUB3ApDod9iciaPuvov3tz7TPYL3xz2sax0ROjCNSMEpQbmPeicxzX7aibxKFOnmO6Qg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```c++
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& candidates, int target,int sum,int startIndex)
    {
        if(sum>target)
            return;
        if(sum==target)
        {
            result.push_back(path);
            return;
        }
        for(int i=startIndex;i<candidates.size();i++)
        {
            sum+=candidates[i];
            path.push_back(candidates[i]);
            backtracking(candidates,target,sum,i);
            sum-=candidates[i];
            path.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtracking(candidates,target,0,0);
        return result;
    }
};
```

### 40、组合总和II

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5v0YcZ1fEmnNP568yOjsbyvXic0aQZqCfpd5WtPuNINVibupaDPIbE1yicHHOfPlUCLLNmBMr2Ol7yw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

本题数组candidates的元素是有重复的，而**39.组合总和**是无重复元素的数组candidates

所以就要考虑去掉重复组合，用一个used数组

```
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& candidates, int target,int sum,int startIndex,vector<bool>&used)
    {
        if(sum>target)
            return;
        if(sum==target)
        {
            result.push_back(path);
            return;
        }
        for(int i=startIndex;i<candidates.size()&&sum+candidates[i]<=target;i++)
        {
            // 要对同一树层使用过的元素进行跳过
            if(i>0&&candidates[i]==candidates[i-1]&&used[i-1]==false)
            {
                continue;
            }
            sum+=candidates[i];
            path.push_back(candidates[i]);
            used[i]=true;
            backtracking(candidates,target,sum,i+1,used);
            sum-=candidates[i];
            path.pop_back();
            used[i]=false;


        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<bool> used(candidates.size(), false);
        sort(candidates.begin(),candidates.end());
        backtracking(candidates,target,0,0,used);
        return result;
    }
};
```

### 131、分割回文串

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv50AUHOlEQOnrbmKPI7VFTy7b0IrXDBicAFYvlpDvq5pAGMf0yFTCJzP8F9VN2J6kDHY0hvalS04Fw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
class Solution {
public:
    vector<vector<string>> result;
    vector<string> path;
    bool ishuiwen(const string& s,int start,int end)
    {
        for(int i=start,j=end;i<j;i++,j--)
        {
            if(s[i]==s[j])
                continue;
            else
                return false;
        }
        return true;
    }
    void backtracking(string s,int startIndex)
    {
        //终止条件
        if(startIndex>=s.size())
        {
            result.push_back(path);
            return;
        }
        
        for(int i=startIndex;i<s.size();i++)
        {
            if(ishuiwen(s,startIndex,i))
            {
                string str=s.substr(startIndex,i-startIndex+1);
                path.push_back(str);
            }
            else
                continue;
            
            backtracking(s,i+1);
            path.pop_back();
        }

    }
    vector<vector<string>> partition(string s) {
        backtracking(s,0);
        return result;

    }
};
```

### 93、复原IP地址

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4mRlmnQrMpYlg0sj07SlxGgBmBJ0vBwicV0l2bdQu9X2xIT9G9qV90azokH0xlyV8wEA1fJTXEYPQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```c++
//在原来字符串上插入小数点
//不在开辟新的空间存储中间结果
class Solution {
public:
    vector<string> result;
    void backtracking(string&s,int startIndex,int pointnum)
    {
        //小数点数
        if(pointnum==3)
        {
            if(is_valid(s,startIndex,s.size()-1))
            {
                result.push_back(s);
            }
            return;	//注意返回的位置
        }
        for(int i=startIndex;i<s.size();i++)
        {
            if(is_valid(s,startIndex,i))
            {
                s.insert(s.begin()+i+1,'.');
                pointnum++;
                backtracking(s,i+2,pointnum); 	//接着走就用i
                pointnum--;
                s.erase(s.begin()+i+1);
            }
            else break;
        }
    }
    vector<string> restoreIpAddresses(string s) {
        backtracking(s,0,0);
        return result;
    }
    //判断start-end字符串是否合法
    bool is_valid(string&s,int start,int end)
    {
        if(start>end)
            return false;
        if(s[start]=='0'&&start!=end)
            return false;
        int num=0;
        for(int i=start;i<=end;i++)
        {
            if(s[i]>'9'||s[i]<'0')
                return false;
            num=num*10+(s[i] - '0');
            if(num>255)
                return false;
        }
        return true;
    }
};
```

### 78、子集

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv5JJBVn1TjdicHP2W7eMd1bU5FrR0uSQT8Juhrjd7MeicTJdNXmia6etDicOOVfFDicPTTP0aERNcYD4yQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```c++
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking(vector<int>& nums,int startIndex)
    {
        result.push_back(path);
        if(startIndex>=nums.size())
        {
            return;
        }
        for(int i=startIndex;i<nums.size();i++)
        {
            path.push_back(nums[i]);
            backtracking(nums,i+1);
            path.pop_back();
        }
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        backtracking(nums, 0);
        return result;
    }
};
```

### 剑指offer 38、字符串的排列

```c++
//vector<string>res为全局变量，表示最终的结果集，最后要返回的
class Solution {
public:
    vector<string>res;
 void backtrack(string s,string& path,vector<bool>& used)//used数组
    {
        if(path.size()==s.size())
        {
            res.push_back(path);
            return;
        }
        for(int i=0;i<s.size();i++)
        {
            if(!used[i])
            {
                if(i>=1&&s[i-1]==s[i]&&!used[i-1])//判重剪枝
                    continue;
                path.push_back(s[i]);
                used[i]=true;
                backtrack(s,path,used);
                used[i]=false;
                path.pop_back();
            }   
        }
    }

vector<string> permutation(string s) {
        if(s.size()==0)
            return{};
        string temp="";
        sort(s.begin(),s.end());
        vector<bool>used(s.size());
        backtrack(s,temp,used);
        return res;
    }
};

```

### 79、单词搜索

```c++
//
class Solution {
public:
    int rows,cols;
    bool dfs(vector<vector<char>>& board, string word,int u,int x,int y)
    {
        if((x<0||x>=rows)||(y<0||y>=cols)||(board[x][y]!=word[u])) return false;
        if(u==word.size()-1) return true;
        char t=board[x][y];
        board[x][y]='*';
        if(dfs(board,word,u+1,x+1,y)||dfs(board,word,u+1,x-1,y)||dfs(board,word,u+1,x,y+1)||dfs(board,word,u+1,x,y-1))
            return true;
        // 如果不通，回溯至上一个状态
        board[x][y]=t;
        return false;
    }
    bool exist(vector<vector<char>>& board, string word) {
        rows=board.size();
        cols=board[0].size();
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                if(dfs(board,word,0,i,j))
                    return true;
            }
        }
        return false;
    }
};

class Solution {
public:
    int n,m;
    int dx[4]={-1,0,1,0},dy[4]={0,1,0,-1};  //左 上 右 下
    bool dfs(vector<vector<char>>& board,int x,int y,string& word,int u)
    {
        if(board[x][y]!=word[u]) return false;
        if(u==word.size()-1) return true;
        //避免重复使用
        board[x][y]='.';
        for(int i=0;i<4;i++)
        {
            int a=x+dx[i],b=y+dy[i];
            if(a>=0&&a<n&&b>=0&&b<m)
            {
                if(dfs(board,a,b,word,u+1))
                    return true;
            }
        }
        //// 如果不通，回溯至上一个状态
        board[x][y]=word[u];  //如果不匹配的话再设置成原来的数，回溯思想
        return false;
    }
    bool exist(vector<vector<char>>& board, string word) {
        if(board.empty()||board[0].empty()) return false;
        n=board.size(),m=board[0].size();
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {
                if(dfs(board,i,j,word,0))
                    return true;
            }
        }
        return false;
    }
};
```

### 47、全排列||

![image-20210812130548551](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210812130548551.png)

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void back(vector<int>& nums,vector<bool>& used)
    {
        if(path.size()==nums.size())
        {
            res.push_back(path);
            return;
        }
        for(int i=0;i<nums.size();i++)
        {
            // used[i - 1] == true，说明同一树支nums[i - 1]使用过
            // used[i - 1] == false，说明同一树层nums[i - 1]使用过
            // 如果同一树层nums[i - 1]使用过则直接跳过
            if(i>0&&nums[i]==nums[i-1]&&used[i-1]==false)
            {
                continue;
            }
            if(used[i]==false)
            {
                used[i]=true;
                path.push_back(nums[i]);
                back(nums,used);
                path.pop_back();
                used[i]=false;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        res.clear();
        sort(nums.begin(),nums.end());  //这样相同的数才能放在一起
        vector<bool> used(nums.size(),false);
        back(nums,used);
        return res;
    }

};
```

### 51、N皇后

![image-20210826150434585](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210826150434585.png)

```c++
class Solution {
public:
    vector<vector<string>> res;
    void backtracing(int n,int row,vector<string>& board)
    {
        if(n==row)
        {
            res.push_back(board);
            return;
        }
        for(int col=0;col<n;col++)
        {
            if(is_valid(row,col,n,board))  //剪枝
            {
                board[row][col]='Q';
                backtracing(n,row+1,board);
                board[row][col]='.';
            }
        }
    }
    bool is_valid(int row,int col,int n,vector<string>& board)
    {
        //检查列
        for(int i=0;i<row;i++)
        {
            if(board[i][col]=='Q')
                return false;
        }
        //检查左上
        for(int i=row-1,j=col-1;i>=0&&j>=0;i--,j--)
        {
            if(board[i][j]=='Q')
                return false;
        }
        //检查右上
        for(int i=row-1,j=col+1;i>=0&&j<n;i--,j++)
        {
            if(board[i][j]=='Q')
                return false;
        }
        return true;
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<string> board(n,string(n,'.'));
        backtracing(n,0,board);
        return res;
    }
};
```



## 贪心算法

### 455、分发饼干

```
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {

        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int index = s.size() - 1; // 饼干数组的下表
        int result = 0;
        for (int i = g.size() - 1; i >= 0; i--) {
            if (index >= 0 && s[index] >= g[i]) {
                result++;
                index--;
            }
        }
        return result;
    }
};
```

### 376、摆动序列

```
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int size=nums.size();
        if(size<=0)
            return size;
        int result=1;
        int pre=0;
        int cur=0;
        for(int i=1;i<size;i++)
        {
            cur=nums[i]-nums[i-1];
            if((cur<0&&pre>=0)||(cur>0&&pre<=0))
            {
                result++;
                pre=cur;

            }
        }
        return result;
    }
};
```

### 53、最大子序和

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int result = INT32_MIN;
        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            count += nums[i];
            if (count > result) { // 取区间累计的最大值（相当于不断确定最大子序终止位置）
                result = count;
            }
            if (count <= 0) count = 0; // 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
        }
        return result;
    }
};

//动态规划
/*
状态定义： 设动态规划列表 dp ，dp[i] 代表以元素 nums[i] 为结尾的连续子数组最大和。
为何定义最大和 dp[i] 中必须包含元素 nums[i] ：保证 dp[i] 递推到 dp[i+1] 的正确性；如果不包含 nums[i] ，递推时则不满足题目的连续子数组要求。
*/
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n=nums.size();
        vector<int> dp(n+1,0);
        dp[0]=nums[0];
        int result=dp[0];
        for(int i=1;i<n;i++)
        {
            dp[i]=max(dp[i-1]+nums[i],nums[i]);
            if(dp[i]>result) result=dp[i];
        }
        return result;

    }
};
```

### 152、乘积最大子数组（DP）

```c++
//负数乘以负数，会变成正数，所以解这题的时候我们需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。
 
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n=nums.size();
        if(n==1)
            return nums[0];
        if(n==0) return 0;
        int res=nums[0];
        int maxp=nums[0];
        int minp=nums[0];
        for(int i=1;i<n;i++)
        {
            int t=maxp;
            maxp=max(max(maxp*nums[i],nums[i]),minp*nums[i]);
            minp=min(min(minp*nums[i],nums[i]),t*nums[i]);
            res=max(maxp,res);
        }
        return res;
    }
};
```



### 121、买卖股票的最佳时机（DP、双指针）

1、当n = k 时，设 cur 保存在 A[k] 卖出条件下的最大利润，profit 保存最终最大利润；
2、当n = k + 1 时，需要计算在 A[k + 1] 卖出下的最大利润。可在步骤 1 的基础上进行，转移关系：cur ← MAX(cur + A[k + 1] - A[k], A[k + 1] - A[k])；简要解释，比大小——前一个数含义是认可 A[1..k] 的买入时机，卖出时机顺延为 A[k + 1]；后一个数代表在 A[k] 买入，在 A[k + 1] 卖出。求 A[k + 1] 卖出下的最大利润唯有此 2 种情况，更新 cur 使之对应为在 A[k + 1] 下卖出的最大利润profit ← MAX(profit, cur)。
遍历数组 A，就可以得到最大利润 profit。

![image-20210304112228728](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210304112228728.png)

```c++
class Solution {
public:
    
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
        if(n<=1) return 0;

        int cur=prices[1]-prices[0];
        int profit=prices[1]-prices[0];
        for(int i=2;i<n;i++)
        {
            int tmp=prices[i]-prices[i-1];
            cur=max(cur+tmp,tmp);
            profit=max(profit,cur);
        }
        return profit>0?profit:0;

    }
};

//二维的动态规划
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len=prices.size();
        if(prices.size()<=1)
            return 0;
        vector<vector<int>> dp(len,vector<int>(2,0));
        
        dp[0][0]=0;
        dp[0][1]=-prices[0];   //今天持股

        for(int i=1;i<len;i++)
        {
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]);
            dp[i][1]=max(dp[i-1][1],-prices[i]);
        }
        return dp[len-1][0];
    }
};
//进行降维的动态规划
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len=prices.size();
        if(prices.size()<=1)
            return 0;
         int nohold=0;
         int hold=-prices[0];
         for(int i=1;i<len;i++)
         {
             nohold=max(nohold,hold+prices[i]);
             hold=max(hold,-prices[i]);
         }
         return nohold;
    }
};
//双指针
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len=prices.size();
        if(prices.size()<=1)
            return 0;
        int min_money=prices[0];   //遍历过的最小值
        int maxmoney=0;            //利润最大值
        for(int i=1;i<len;i++)
        {
            min_money=min(min_money,prices[i]);
            maxmoney=max(maxmoney,prices[i]-min_money);
        }
        return maxmoney;
    }
};
//单调栈
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        stack<int> st;
        int res;
        for(int i=0;i<prices.size();i++)
        {
            if(st.empty()||prices[i]<st.top())
            {
                st.push(prices[i]);
            }
            else
            {
                res=max(res,(prices[i]-st.top()));
            }
        }
        return res;
    }
};
```

### 309、买卖股票的最佳时期（含冷冻期）

![image-20210405101418657](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210405101418657.png)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
         vector<vector<int>> dp(n+1,vector<int>(2,0));
         if(n<=1) return 0;
         dp[0][0] = 0;
         dp[0][1] = 0;
         dp[1][0]=0;
         dp[1][1]=-prices[0];
         for(int i=2;i<=n;i++)
         {
             dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i-1]);
             dp[i][1]=max(dp[i-1][1],dp[i-2][0]-prices[i-1]);
         }
         return dp[n][0];
            
    }
};
//优化空间

```

### 122、买卖股票的最佳时机II（DP）

![image-20210405095247396](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210405095247396.png)

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int cur=0;
        int result=0;
        for(int i=1;i<prices.size();i++)
        {
            cur=prices[i]-prices[i-1];
            if(cur>0)
            {
                result+=cur;
            }

        }
        return result;
    }
};
//动态规划
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len=prices.size();
        if(prices.size()<=1)
            return 0;
        vector<vector<int>> dp(len,vector<int>(2,0));
        
        dp[0][0]=0;
        dp[0][1]=-prices[0];   //今天持股

        for(int i=1;i<len;i++)
        {
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]);  //今天不持股票
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i]);  //今天持有股票，第二种情况：加上前一天的利润再进行买入
        }
        return dp[len-1][0];
    }
};
```

### 714、买卖股票的最佳时机（含手续费）

- 情况一：收获利润的这一天并不是收获利润区间里的最后一天（不是真正的卖出，相当于持有股票），所以后面要继续收获利润。
- 情况二：前一天是收获利润区间里的最后一天（相当于真正的卖出了），今天要重新记录最小价格了。
- 情况三：不作操作，保持原有状态（买入，卖出，不买不卖）

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int result=0;
        int minprice=prices[0];
        for(int i=1;i<prices.size();i++)
        {
            if(prices[i]<minprice)
                minprice=prices[i];
            if(prices[i]>minprice&&prices[i]<=minprice+fee)
                continue;   //情况3
            if(prices[i]>minprice+fee)
            {
                result+=prices[i]-minprice-fee;
                minprice=prices[i]-fee;   //情况1
            }
        }
        return result;
    }
};
//动态规划
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int len=prices.size();
        if(prices.size()<=1)
            return 0;
        vector<vector<int>> dp(len,vector<int>(2,0));
        
        dp[0][0]=0;
        dp[0][1]=-prices[0]-fee;   //今天持股

        for(int i=1;i<len;i++)
        {
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]);  //今天不持股票
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i]-fee);//买入的时候利润-手续费  
        }
        return dp[len-1][0];
    }
};
```



### 55、跳跃游戏

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int cover=0;
        if(nums.size()==1) return true;
        for(int i=0;i<=cover;i++)
        {
            cover=max(i+nums[i],cover);
            if(cover>=nums.size()-1) return true;
        }
        return false;
    }
};
```

### 45、跳跃游戏II

```
class Solution {
public:
    int jump(vector<int>& nums) {
        
        int maxdistence=0;
        
        int end=0;
        int result=0;

        for(int i=0;i<nums.size()-1;i++)
        {
            maxdistence=max(i+nums[i],maxdistence);
            if(i==end)
            {
                result++;
                end=maxdistence;
            }
        }
        return result;

    }
};
```

### 1005、K次取反后最大化的数组和

```
class Solution {
public:
    static bool cmp(int a,int b)
    {
        return abs(a)>abs(b);    //从大到小排序
    }
    int largestSumAfterKNegations(vector<int>& A, int K) {
        int sum=0;

        sort(A.begin(),A.end(),cmp);
        for(int i=0;i<A.size();i++)
        {
            if(A[i]<0&&K>0)
            {
                A[i]=-A[i];
                K--;
            }
        }
        while(K--) 
            A[A.size()-1]=-A[A.size()-1];

        for(int a:A)
            sum+=a;
        return sum;


    }
};
```

### 134、加油站

```c++
//通过三个变量来计算，totalsum是计算总的可不可以走完路程，cursum是计算当前的可不可以走
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int start=0;
        int totalSum=0;
        for(int i=0;i<gas.size();i++)
        {
            cursum+=gas[i]-cost[i];
            totalSum+=gas[i]-cost[i];
            if(cursum<0)
            {
                cursum=0;
                start=i+1;
            }

        }
        if(totalSum<0) return -1;
        return start;
    }
};
```

### 860、柠檬水找零

```
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five=0;
        int ten=0;
        int twenty=0;
        for(int bill:bills)
        {
            if(bill==5)
            {
                five++;
            }
            if(bill==10)
            {
                if(five<=0) return false;
                ten++;
                five--;
            }
            if(bill==20)
            {
                if(five>0&&ten>0)
                {
                    twenty++;
                    five--;
                    ten--;
                }
                else if(five>=3)
                {
                    five-=3;
                    twenty++;
                }
                else return false;
            }
        }
        return true;
    }
};
```

### 406、根据身高重建队列

按照身高从大到小排列，然后再根据k值插入

```
class Solution {
public:
    static bool cmp(const vector<int>& a,const vector<int>& b)
    {
        if(a[0]==b[0]) return a[1]<b[1];
        return a[0]>b[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(),people.end(),cmp);
        list<vector<int>> que;
        
        for(int i=0;i<people.size();i++)
        {
            int position=people[i][1];
            std::list<vector<int>>::iterator it=que.begin();
            while(position--)
            {
                it++;
            }
            que.insert(it,people[i]);
        }
        return vector<vector<int>>(que.begin(),que.end());
    }
};
```

### 452、用最少数量的箭引爆气球

```
//寻找重叠气球最小右边界
class Solution {
public:
    static bool cmp(const vector<int>&a,const vector<int>&b)
    {
        return a[0]<b[0];
    }
    int findMinArrowShots(vector<vector<int>>& points) {
        if(points.size()==0) return 0;
        sort(points.begin(),points.end(),cmp);

        int result=1;
        for(int i=1;i<points.size();i++)
        {
            if(points[i][0]>points[i-1][1])
            {
                result++;
            }
            else
            {
                points[i][1]=min(points[i][1],points[i-1][1]);
            }
        }
        return result;
    }
};
```

### 435、无重叠区间

```
class Solution {
public:
    static bool cmp(const vector<int>&a,const vector<int>&b)
    {
        return a[1]<b[1];
    }
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        if(intervals.size()==0) return 0;
        sort(intervals.begin(),intervals.end(),cmp);
        int count=1;  //非交叉区间个数
        int end=intervals[0][1];
        for(int i=1;i<intervals.size();i++)
        {
            if(intervals[i][0]>=end)
            {
                count++;
                end=intervals[i][1];
            }
        }
        return intervals.size()-count;
    }
};
```

### 763、划分字母区间

```
class Solution {
public:
    vector<int> partitionLabels(string S) {
        int hash[27]={0};
        for(int i=0;i<S.size();i++)  // 统计每一个字符最后出现的位置
        {
            hash[S[i]-'a']=i;
        }
        vector<int> result;
        int right=0;
        int left=0;
        for(int i=0;i<S.size();i++)
        {
            right=max(right,hash[S[i]-'a']);  // 找到字符出现的最远边界
            if(i==right)
            {
                result.push_back(right-left+1);
                left=i+1;
            }
        }
        return result;
    }
};
```

### 56、合并区间

```c++
//先按照起始大小从小到大进行排序
//再比较第二个值和下一个第一个值的大小，进行合并区间
class Solution {
public:
     static bool cmp (const vector<int>& a, const vector<int>& b) {
        return a[0] < b[0];
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> result;
        if(intervals.size()==0) return result;
        sort(intervals.begin(),intervals.end(),cmp);

        bool flag=false;
        int length=intervals.size();
        for(int i=1;i<length;i++)
        {
            int start=intervals[i-1][0];
            int end=intervals[i-1][1];
            while(i<length&&intervals[i][0]<=end)
            {
                
                end=max(end,intervals[i][1]);
                if(i==length-1) flag=true;
                i++;
            }
            result.push_back({start,end});
        }
        if(flag==false)
        {
            result.push_back({intervals[length-1][0],intervals[length-1][1]});
        }
        return result;
    }
};
//解法二
class Solution {
public:
    static bool cmp(const vector<int>&a,const vector<int>&b)
    {
        if(a[0]!=b[0])
            return a[0]<b[0];
        return a[1]<b[1];
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> res;
        sort(intervals.begin(),intervals.end(),cmp);
        int n=intervals.size();
        if(n==0) return res;
        vector<int> cur = intervals[0];
        for(int i=1;i<n;i++)
        {
            if(cur[1]<intervals[i][0])
            {
                res.push_back(cur);
                cur=intervals[i];
            }else if(cur[1]<intervals[i][1])
            {
                cur[1]=intervals[i][1];
            }
        }
        res.push_back(cur);
        return res;
    }
};
```

### 738、单调递增的数字

解题思路：从后向前遍历

```
class Solution {
public:
    int monotoneIncreasingDigits(int N) {
        string strnum=to_string(N);

        int flag=strnum.size();
        for(int i=strnum.size()-1;i>0;i--)
        {
            if(strnum[i-1]>strnum[i])
            {
                flag=i;
                strnum[i-1]--;
            }
        }
        for(int i=flag;i<strnum.size();i++)
        {
            strnum[i]='9';
        }
        return stoi(strnum);
    }
};
```

### 135、分发糖果

- **规则定义：** 设学生 A*A* 和学生 B*B* 左右相邻，A*A* 在 B*B* 左边；
  - **左规则：** 当 ratings_B>ratings_A*r**a**t**i**n**g**s**B*>*r**a**t**i**n**g**s**A*时，B*B* 的糖比 A*A* 的糖数量多。
  - **右规则：** 当 ratings_A>ratings_B*r**a**t**i**n**g**s**A*>*r**a**t**i**n**g**s**B*时，A*A* 的糖比 B*B* 的糖数量多。

![image-20210822131658922](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210822131658922.png)

```c++
class Solution {
public:
    int candy(vector<int>& ratings) {
        int n=ratings.size();
        vector<int> left(n,1);
        vector<int> right=left;
        for(int i=1;i<n;i++)
        {
            if(ratings[i]>ratings[i-1])
                left[i]=left[i-1]+1;
        }
        for(int i=n-2;i>=0;i--)
        {
            if(ratings[i]>ratings[i+1])
                right[i]=right[i+1]+1;
        }
        int res=0;
        for(int i=0;i<n;i++)
        {
            left[i]=max(left[i],right[i]);
            res+=left[i];
        }
        return res;
    }
};
```



# 数据结构相关

## 数组

### 35、搜索插入的位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

```c++
方法一：暴力解法
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
       for (int i = 0; i < nums.size(); i++) {
       
            if (nums[i] >= target) { // 一旦发现大于或者等于target的num[i]，那么i就是我们要的结果
                return i;
            }
        }
        // 目标值在数组所有元素之后的情况 
        return nums.size(); // 如果target是最大的，或者 nums为空，则返回nums的长度
    }
};
//方法二：二分查找[left,right]
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
       int left=0;
       int right=nums.size()-1;
       while(left<=right)
       {
           int middle=left+(right-left)/2;
           if(nums[middle]>target)
                right=middle-1;
            else if(nums[middle]<target)
                left=middle+1;
            else
                return middle;
       }
       return right+1;
    }
};
```

### 27、移除元素(双指针)

给你一个数组 *nums* 和一个值 *val*，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 *val* 的元素，并返回移除后数组的新长度。

```
//暴力解法
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int size=nums.size();
        for(int i=0;i<size;i++)
        {
            if(nums[i]==val)
            {
                for(int j=i+1;j<size;j++)
                {
                    nums[j-1]=nums[j];
                }
            i--;
            size--;
            }
            
        }
        
        return size;
    }
};

//双指针解法
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slowIndex = 0; 
        for (int fastIndex = 0; fastIndex < nums.size(); fastIndex++) {  
            if (val != nums[fastIndex]) { 
                nums[slowIndex] = nums[fastIndex]; 
                slowIndex++；
            }
        }
        return slowIndex;
    }
};
```

### 209、长度最小的子数组(滑动窗口)

所谓滑动窗口，**「就是不断的调节子序列的起始位置和终止位置，从而得出我们要想的结果」**。

```c++
//暴力解法
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int result=INT32_MAX;
        int count=0;
        int j=0;
        vector<int> vec;
        //vec.push_back(0);
        for(j=0;j<nums.size();j++)
        {
            int sum=0;
            for(int i=j;i<nums.size();i++)  //先从下标0开始找起
            {
                sum+=nums[i];
                if(sum>=s)
                {
                    count=i-j+1;
                    result=result < length ? result : length;
                    break;
                }

            }
        }
         return result == INT32_MAX ? 0 : result;
        
        
    }
};
//滑动窗口
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int result = INT32_MAX;
        int sum = 0; // 滑动窗口数值之和
        int i = 0; // 滑动窗口起始位置
        int subLength = 0; // 滑动窗口的长度
        for (int j = 0; j < nums.size(); j++) {
            sum += nums[j];
            // 注意这里使用while，每次更新 i（起始位置），并不断比较子序列是否符合条件
            while (sum >= s) {
                subLength = (j - i + 1); // 取子序列的长度
                result = result < subLength ? result : subLength;
                sum -= nums[i++]; // 这里体现出滑动窗口的精髓之处，不断变更i（子序列的起始位置）
                //比如：1 1 1 1 1 5  和7
       
            }
        }
        // 如果result没有被赋值的话，就返回0，说明没有符合条件的子序列
        return result == INT32_MAX ? 0 : result;
    }
};
```

### 189、旋转数组

```c++
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k%=nums.size();
        reverse(nums.begin(),nums.end());
        reverse(nums.begin(),nums.begin()+k);
        reverse(nums.begin()+k,nums.end());
    }
};
```



## 字符串

**「其实很多数组填充类的问题，都可以先预先给数组扩容带填充后的大小，然后在从后向前进行操作。」**

### 344、反转字符串

```
//方法一：利用库函数
class Solution {
public:
    void reverseString(vector<char>& s) {
        reverse(s.begin(), s.end());
    }
};
//方法二：
class Solution {
public:

    void reverseString(vector<char>& s) {
        int m=s.size()-1;
        for(int i=0;i<s.size()/2;i++)
        {
            swap(s[i],s[m-i]);
        }
    }
};
```

### 剑指offer-05：替换空格

```
//使用了额外空间
class Solution {
public:
    string replaceSpace(string s) {
        string s1="";

        for(int i=0;i<s.size();i++)
        {
            if(s[i]!=' ')
            {
                s1=s1+s[i];
            }
            else
            {
                s1=s1+"%20";
            }
           
        }
        return s1;
    }
};
//双指针法
//i指向新长度的末尾，j指向旧长度的末尾。
class Solution {
public:
    string replaceSpace(string s) {
        int count = 0; // 统计空格的个数
        int sOldSize = s.size();
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') {
                count++;
            }
        }
        // 扩充字符串s的大小，也就是每个空格替换成"%20"之后的大小
        s.resize(s.size() + count * 2);
        int sNewSize = s.size();
        // 从后先前将空格替换为"%20"
        for (int i = sNewSize - 1, j = sOldSize - 1; j < i; i--, j--) {
            if (s[j] != ' ') {
                s[i] = s[j];
            } else {
                s[i] = '0';
                s[i - 1] = '2';
                s[i - 2] = '%';
                i -= 2;
            }
        }
        return s;
    }
};
```

### 151、反转字符串里的单词

![image-20201128150124941](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20201128150124941.png)

```c++
class Solution {
public:
    string reverseWords(string s) {
       int k=0;  //新的单词的起始下标
       for(int i=0;i<s.size();i++)
       {
           if(s[i]==' ') continue;
           int j=i;   //j表示不是空格之后首字母下标
           int t=k;   //移去空格的新下标
           while(j<s.size()&&s[j]!=' ')
           {
               s[t++]=s[j++];
           }
           reverse(s.begin()+k,s.begin()+t);
           s[t++]=' ';
           k=t;  //更新下标，变成下一个字母的起始下标·
           i=j;  //跳过字母
       }
       if(k) k--;  //最后多余的空格,因为这是在s的基础上翻转的，最后肯定剩下空格
       s.erase(s.begin()+k,s.end());
       reverse(s.begin(),s.end());
       return s;
    }
};

```

### 557、反转字符串中的单词|||

```c++
class Solution {
public:
    string reverseWords(string s) {
        int k=0;
        for(int i=0;i<s.size();i++)
        {
            if(s[i]==' ') continue;
            int j=i;  //有效单词的下标
            int t=k;
            while(j<s.size()&&s[j]!=' ')
            {
                s[t++]=s[j++];
            }
            //反转+空格
            reverse(s.begin()+k,s.begin()+t);
            s[t++]=' ';
            k=t;
            i=j;
            
        }
        if(k) k--;
        s.erase(s.begin()+k,s.end());

        return s;
    }
};
```

### 415、字符串相加

```c++
class Solution {
public:
    string addStrings(string num1, string num2) {
        string res;
        int t=0,n=num1.size()-1,m=num2.size()-1;
        while(n>=0||m>=0)
        {
            if(n>=0)
            {
                t+=num1[n--]-'0';
            }
            if(m>=0)
            {
                t+=num2[m--]-'0';
            }
            res=to_string(t%10)+res;
            t=t/10;
        }
        if(t)
        {
            res='1'+res;
        }
        return res;
    }
};
//思路：先补齐，再从后向前加
class Solution {
public:
    string addStrings(string num1, string num2) {
        int s1=num1.size();
        int s2=num2.size();
        while(s1>s2)
        {
            num2="0"+num2;
            s2++;
        }
        while(s1<s2)
        {
            num1="0"+num1;
            s1++;
        }
        char flag='0';
        char s;
        string res="";
        for(int i=s1-1;i>=0;i--)
        {
            int n=num1[i]-'0'+num2[i]-'0'+flag-'0';
            if(n>=10)
            {
                flag='1';
                n=n%10;
            }
            else
                flag='0';   
            s=n+'0';
            res=s+res;
        }
        if(flag=='1')
            res='1'+res;
        return res;
       
    }
};
```

### 445、两数相加||

```c++
//利用反转链表来做
class Solution {
public:
    ListNode* reverse(ListNode* head)
    {
        ListNode* pre=nullptr;
        ListNode* cur=head;
        while(cur)
        {
            ListNode* tmp=cur->next;
            cur->next=pre;
            pre=cur;
            cur=tmp;
        }
        return pre;
    }
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* dummy=new ListNode(-1);
        ListNode* cur=dummy;
        ListNode* l11=reverse(l1);
        ListNode* l22=reverse(l2);
        int t=0;
        while(l11||l22)
        {
            if(l11)
            {
                t+=l11->val;
                l11=l11->next;
            }
            if(l22)
            {
                t+=l22->val;
                l22=l22->next;
            }
            cur->next=new ListNode(t%10);
            t=t/10;
            cur=cur->next;                 
        }
        if(t)
        {
            cur->next=new ListNode(1);
        }
        ListNode* res=dummy->next;
        return reverse(res);
    }
};
//利用栈来做。
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        stack<int> s1,s2,res;
        while(l1) 
        {
            s1.push(l1->val);
            l1=l1->next;
        }
        while(l2) 
        {
            s2.push(l2->val);
            l2=l2->next;
        }
        int t=0;
        while(!s1.empty()||!(s2.empty()))
        {
            if(!s1.empty()) 
            {
                t+=s1.top();
                s1.pop();
            }
            if(!s2.empty())
            {
                t+=s2.top();
                s2.pop();
            }
            res.push(t%10);
            t=t/10;
        }
        if(t==1) res.push(1);
        ListNode* dummy=new ListNode(-1);
        ListNode* cur=dummy;
        while(!res.empty())
        {
            cur->next=new ListNode(res.top());
            res.pop();
            cur=cur->next;
        }
        return dummy->next;
    }
};
```



### 443、压缩字符串

```c++
/*
三指针
rear:用来指压缩之后的下标-1
left\right用来判断字符是否相等
*/
class Solution {
public:
    int compress(vector<char>& chars) {
        int rear=-1;
        int left=0;
        int right=0;
        while(right<chars.size())
        {
            left=right;
            while(right<chars.size()&&chars[left]==chars[right]) right++;
            chars[++rear]=chars[left];
            if(right-left==1) continue;
            for(auto x:to_string(right-left))
                chars[++rear]=x;
        }
        return rear+1;
    }
};
```



### 28. 实现 strStr()---KMP

```

class Solution {
public:
    void getNext(int *next,const string &s)  //填充next数组
    {
        int j=-1;
        next[0]=-1;
        for(int i=1;i<s.size();i++)
        {
            while(j>=0&&s[i]!=s[j+1])
            {
                j=next[j];
            }
            if(s[i]==s[j+1])
            {
                j++;
            }
            next[i]=j;
        }
    }
    int strStr(string haystack, string needle) {
        if(needle.size()==0)
            return 0;
        int next[needle.size()];
        getNext(next,needle);
        int j = -1; // // 因为next数组里记录的起始位置为-1
        for (int i = 0; i < haystack.size(); i++) { // 注意i就从0开始
            while(j >= 0 && haystack[i] != needle[j + 1]) { // 不匹配
                j = next[j]; // j 寻找之前匹配的位置
            }
            if (haystack[i] == needle[j + 1]) { // 匹配，j和i同时向后移动 
                j++; 
            }
            if (j == (needle.size() - 1) ) { // 文本串s里出现了模式串t
                return (i - needle.size() + 1); 
                //i代表长的，j代表短的
            }
        }
        return -1;
    }
};
```

### 459、重复的字符串---KMP

```c++
class Solution {
public:
    void getnext(int *next,const string& s)  //生成next数组
    {
        //i:代表后缀末尾，j代表前缀末尾并且还代表i和i之前的最长相等前后缀的长度
        
        int j=-1;
        next[0]=-1;
        for(int i=1;i<s.size();i++)
        {
            while(j>=0&&s[i]!=s[j+1])
            {
                j=next[j];
            }
            if(s[i]==s[j+1])
            {
                j++;   //i和i之前的最长相等前后缀的长度+1
            }
            next[i]=j;
        }
    }
    bool repeatedSubstringPattern(string s) {
        if(s.size()==0)
        {
            return false;
        }
        int next[s.size()];
        getnext(next,s);
        int len=s.size();
        if(next[len-1]!=-1&&len%(len-(next[len-1]+1))==0)
        	//len-(next[len-1]+1)--代表重复的子字符串的长度
            return true;
        else
            return false;
    }
};
```

### 49、字母异位词分组

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        int n=strs.size();
        unordered_map<string,vector<string>> map;
        for(int i=0;i<n;i++)
        {
            string key=strs[i];
            sort(key.begin(),key.end());
            map[key].push_back(strs[i]);
        }
        vector<vector<string>> res;
        for(auto i:map)
        {
            res.push_back(i.second);
        }
        return res;
    }
};
```

### 165、比较版本号

```c++
class Solution {
public:
    int compareVersion(string v1, string v2) {
        int i=0,j=0;  //实际的一直往前走的指针
        while(i<v1.size()||j<v2.size())
        {
            int x=i,y=j;
            while(x<v1.size()&&v1[x]!='.') x++;
            while(y<v2.size()&&v2[y]!='.') y++;
            int a=x==i?0:atoi(v1.substr(i,x-i).c_str());
            int b=y==j?0:atoi(v2.substr(j,y-j).c_str());
            if(a<b) return -1;
            if(a>b) return 1;
            i=x+1;  //注意：不能是x++
            j=y+1;
        }
        return 0;
    }
};

```

### 929、独特的电子邮件地址

```c++
class Solution {
public:
    int numUniqueEmails(vector<string>& emails) {
        unordered_set<string> hash;
        for(auto email:emails)
        {
            int at=email.find('@');
            string name;
            for(auto c:email.substr(0,at))
            {
                if(c=='+') break;
                else if(c!='.') name+=c;
                
            }
            string domain=email.substr(at+1);
            hash.insert(name+'@'+domain);
        }
        return hash.size();
    }
};
```



## 栈和队列

### 232、用栈实现队列

```c++
class MyQueue {
public:
    stack<int> stIn;
    stack<int> stOut;
    /** Initialize your data structure here. */
    MyQueue() {

    }
    /** Push element x to the back of queue. */
    void push(int x) {
        stIn.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        // 只有当stOut为空的时候，再从stIn里导入数据（导入stIn全部数据）
        if (stOut.empty()) {
            // 从stIn导入数据直到stIn为空
            while(!stIn.empty()) {
                stOut.push(stIn.top());
                stIn.pop();
            }
        }
        int result = stOut.top();
        stOut.pop();
        return result;
    }

    /** Get the front element. */
    int peek() {
        int res = this->pop(); // 直接使用已有的pop函数
        stOut.push(res); // 因为pop函数弹出了元素res，所以再添加回去
        return res;
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return stIn.empty() && stOut.empty();
    }
};

```

### 225、用队列实现栈

```c++
class MyStack {
public:
    queue<int> que1;
    queue<int> que2; 
    /** Initialize your data structure here. */
    MyStack() {

    }
    
    /** Push element x onto stack. */
    void push(int x) {
        que1.push(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int size=que1.size();
        size--;   //保证while循环执行size-1次
        while(size)
        {
            que2.push(que1.front());
            que1.pop();
            size--;
        }
        int result = que1.front(); // 留下的最后一个元素就是要返回的值
        que1.pop();
        que1 = que2;            // 再将que2赋值给que1
        while (!que2.empty()) { // 清空que2
            que2.pop();
        }
        return result;
    }
    
    /** Get the top element. */
    int top() {
        return que1.back();
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return que1.empty();
    }
};
 
```

### 20、有效的括号

```c++
//标准用栈来解决
class Solution {
public:
    bool isValid(string s) {
       stack<int> st;
       for(int i=0;i<s.size();i++)
       {
           if(s[i]=='(') st.push(')');
           else if(s[i]=='{') st.push('}');
           else if(s[i]=='[') st.push(']');
           else if(st.empty()||s[i]!=st.top()) return false;
           else st.pop();
       }
       return st.empty();
    }
};
```

### 32、最长有效括号(双指针)

![image-20210505184650768](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210505184650768.png)

```c++
//(:1 ):-1
//括号序列合法：所有前缀和>=0,且总和=0
class Solution {
public:
    int work(string s)
    {
        int res=0;
        int count=0;
        for(int i=0,start=0;i<s.size();i++)
        {
            if(s[i]=='(')  count++;
            else
            {
                count--;
                if(count<0) start=i+1,count=0;
                else if(count==0) res=max(res,(i-start+1));
            }
        }
        return res;
    }
    int longestValidParentheses(string s) {
        int res=work(s);
        reverse(s.begin(),s.end());
        for(auto& c:s) c^=1;  
        return max(res,work(s));  //防止(((())
    }
};

//利用栈
class Solution {
public:
    int longestValidParentheses(string s) {
        int res = 0,n = s.length();
        stack<int> st;
        st.push(-1);
        for(int i = 0 ; i < n ; i ++)
        {
            if(s[i] == '(') st.push(i);
            else
            {
                st.pop();
                if(st.empty()) st.push(i);
                else res = max(res,i - st.top());
            }
        }
        return res;
    }
};
```



### 1047、删除字符串所有相邻重复项

```c++
class Solution {
public:
    string removeDuplicates(string S) {
        stack<int> st;
      
        for(int i=0;i<S.size();i++)
        {
            if(st.empty()||S[i]!=st.top())  
                st.push(S[i]);
            else
                st.pop();
        }
        string result = "";
        while (!st.empty()) { // 将栈中元素放到result字符串汇总
            result += st.top();
            st.pop();
        }
        reverse (result.begin(), result.end()); // 此时字符串需要反转一下
        return result;
    }
};
```

### 150、逆波兰表达式

```
//遇到数字就入栈，如果遇到字符就把栈的前两个数字进行相应的运算，然后再入栈，接着做
//st.push(stoi(tokens[i]));  字符串转换成数字
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> st;
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
                int num1 = st.top();
                st.pop();
                int num2 = st.top();
                st.pop();
                if (tokens[i] == "+") st.push(num2 + num1);
                if (tokens[i] == "-") st.push(num2 - num1);
                if (tokens[i] == "*") st.push(num2 * num1);
                if (tokens[i] == "/") st.push(num2 / num1);
            } else {
                st.push(stoi(tokens[i]));
            }
        }
        int result = st.top();
        st.pop(); // 把栈里最后一个元素弹出（其实不弹出也没事）
        return result;
    }
};
//第二种解法
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> st;
        for(int i=0;i<tokens.size();i++)
        {
            if(tokens[i]!="+"&&tokens[i]!="-"&&tokens[i]!="*"&&tokens[i]!="/")
            {
                st.push(stoi(tokens[i]));
            }
            else
            {
               int num1 = st.top();
                st.pop();
                int num2 = st.top();
                st.pop();
                if (tokens[i] == "+") st.push(num2 + num1);
                if (tokens[i] == "-") st.push(num2 - num1);
                if (tokens[i] == "*") st.push(num2 * num1);
                if (tokens[i] == "/") st.push(num2 / num1);
               
            }
        }
        int s=st.top();
       
        return s;
    }
};
```

### 239、滑动窗口最大值（单调队列）

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
       vector<int> result;
       myqueue my_queue;  //创建了一个对象
       for(int i=0;i<k;i++)   //前k个不需要pop
       {
           my_queue.push(nums[i]); 
       }
       result.push_back(my_queue.front());  //装入了第一个窗口的最大值
       for(int i=k;i<nums.size();i++)
       {
           my_queue.pop(nums[i-k]); //慢慢的移动窗口
           my_queue.push(nums[i]);
           result.push_back(my_queue.front());
       }
       return result;

    }

private:
class myqueue
{
public:
    deque<int> que;
    //pop函数的意思是：如果现在队列中的最大值正好是滑动窗口移出的值，就需要pop
    void pop(int val)
    {
        if(!que.empty()&&val==que.front())
            que.pop_front();
    }
    void push(int val)
    {
        while(!que.empty()&&val>que.back())//一直后队列中的最后一个数字进行比较，如果大于最后一个数字，就把队列中最后一个数字删除，小于则加上
        {
            que.pop_back();
        }
        que.push_back(val);
    }
    int front(void)
    {
        return que.front();
    }

};
};
```

### 84、柱状图中的最大矩形（单调栈）

- 此题的本质时找到每个柱形条左边和右边最近的比自己低的柱形条，然后用宽度乘上当前柱形条的高度作为备选方案
- 维护一个单调递增的栈，如果当前柱形条i的高度比栈顶的要低，则栈顶元素cur出栈，出栈后，cur右边第一个比他低的柱形条就是i，左边第一个比他低的就是当前中的top，不断出栈直到栈为空或者柱形条i不再比top低
- 满足2之后，当前柱形条i进栈

```c++
//这样想：如果是一个单增数列，怎么求他的最大值，宽度为1就是最右边一个柱形条，宽度为2就是最右边的两个柱形条（高度就是倒数第二个），依次计算，这样我们就算出了每个宽度下的最大值，所以就可以得到最后的最大值
//以当前高度为h的最大矩形
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size(), ans = 0;
        heights.push_back(-1);
        // 为了算法书写方便，在数组末尾添加高度 -1
        // 这会使得栈中所有数字在最后出栈。

        stack<int> st;
        for (int i = 0; i <= n; i++) {
            while (!st.empty() && heights[i] < heights[st.top()]) {
                int cur = st.top();
                st.pop();

                if (st.empty())
                    ans = max(ans, heights[cur] * i);
                else
                    ans = max(ans, heights[cur] 
                            * (i - st.top() - 1));
            }
            st.push(i);
        }

        return ans;
 
    }
};
```

### 85、最大矩形（单调递增栈）

![image-20210823154250713](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210823154250713.png)

```c++
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int n=matrix.size();
        if(n==0) return 0;
        int m=matrix[0].size();
        int res=0;
         

        vector<int> height(m+1,0);
        height[m]=-0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {
                if(matrix[i][j]=='0')
                    height[j]=0;
                else height[j]++;
            }
            //注意是一行一更新
            stack<int> st;
            for(int j=0;j<=m;j++)
            {
                while(!st.empty()&&height[j]<height[st.top()])
                {
                    int cur=st.top();
                    st.pop();
                    if(st.empty())
                    {
                        res=max(res,height[cur]*j);
                    }
                    else
                        res=max(res,height[cur]*(j-st.top()-1));
                } 
                st.push(j);
            }
        }
        return res;

    }
};
```



### 42、接雨水（单调栈）

```c++
//栈存的是下标
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size(), ans = 0;
        stack<int> st;
        for (int i = 0; i < n; i++) {
            while (!st.empty() && height[st.top()] <= height[i]) {
                int top = st.top();
                st.pop();
                if (st.empty()) break;
                ans += (i - st.top() - 1) 
                        * (min(height[st.top()], height[i]) - height[top]);
            }
            st.push(i);
        }
        return ans;
    }
};
 
```



### 347、前K个高频元素（最小堆）

```c++
// 时间复杂度：O(nlogk)
// 空间复杂度：O(n)
class Solution {
public:
    // 小顶堆
    // //定义仿函数，因为默认的优先队列是大顶堆，我们要构建小顶堆,比较小顶堆的第二个元素的大小
    class mycomparison {
    public:
        bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
            return lhs.second > rhs.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 要统计元素出现频率
        unordered_map<int, int> map; // map<nums[i],对应出现的次数>
        for (int i = 0; i < nums.size(); i++) {
            map[nums[i]]++;   //key是nums的值，val是次数
        }

        // 对频率排序
        // 定义一个小顶堆，大小为k
        priority_queue<pair<int, int>, vector<pair<int, int>>, mycomparison> pri_que;
        
        // 用固定大小为k的小顶堆，扫面所有频率的数值 
        for (unordered_map<int, int>::iterator it = map.begin(); it != map.end(); it++) {
            pri_que.push(*it);
            if (pri_que.size() > k) { // 如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                pri_que.pop();
            }
        }

        // 找出前K个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        vector<int> result(k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pri_que.top().first;
            pri_que.pop();
        }
        return result;

    }
};
```

### 215、数组中的第K个最大元素（最小堆）

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        sort(nums.rbegin(),nums.rend());
        return nums[k-1];
    }
};
//最小堆
class Solution {
public:
    //维护一个K的最小堆
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int,vector<int>,greater<int>> que;
        for(int i=0;i<nums.size();i++)
        {
            que.push(nums[i]);
            if(que.size()>k)
            {
                que.pop();
            }
        }
        return que.top();
    }
};
```

### 295、数据流的中位数(大顶堆、小顶堆)

![image-20210304112850191](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210304112850191.png)

```c++
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int, vector<int>, greater<int> > large;
       priority_queue<int, vector<int>, less<int> > small;
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        if(small.size()>=large.size())
        {
            small.push(num);
            large.push(small.top());
            small.pop();
        }
        else
        {
            large.push(num);
            small.push(large.top());
            large.pop();
        }
    }
    
    double findMedian() {
      
       if(large.size()>small.size())
       {
           return large.top();
       }
       else if(large.size()<small.size())
       {
           return small.top();
       }
       return  (large.top()+small.top())/2.0;
    }
};
```

### 155、最小栈（辅助栈）

```c++
class MinStack {
private:
    stack<int> min_s;
    stack<int> s;
public:
    /** initialize your data structure here. */
    MinStack() {
         min_s.push(INT_MAX);
    }
    
    void push(int x) {
        s.push(x);
        min_s.push(min(min_s.top(),x));
    }
    
    void pop() {
        s.pop();
        min_s.pop();
    }
    
    int top() {
        return s.top();
    }
    
    int getMin() {
        return min_s.top();
    }
};
```

### 394、字符串解码

```c++
//如3[a2[c]b] 使用一次分配律-> 3[accb] 再使用一次分配律->accbaccbaccb
class Solution {
public:
    stack<int> nums;
    stack<string> strs;
    string decodeString(string s) {
        string res="";
        int n=s.size();
        int num=0;
        for(int i=0;i<n;i++)
        {
            if(s[i]>='0'&&s[i]<='9')
            {
                num=num*10+(s[i]-'0');
            }
            else if((s[i]>='a'&&s[i]<='z')||(s[i]>='A'&&s[i]<='Z'))
            {
                res+=s[i];
            }
            else if(s[i]=='[')
            {
                nums.push(num);
                num=0;
                strs.push(res);
                res="";
            }
            else
            {
                int times=nums.top();
                nums.pop();
                for(int i=0;i<times;i++)
                {
                    strs.top()+=res;
                }
                res=strs.top();
                strs.pop();
            }
        }
        return res;
    }
};
```

### 402、移掉K位数字(非递减栈)

```c++
/*
1、构造非递减栈，从左到右遍历数字，如果该数字小于栈顶元素，则栈顶元素出栈，k--;
2、如果K不等于0，则一直弹出栈顶元素直到栈里面的元素=stk.size()-k;
3、将栈里面的元素出栈再reverse
*/
class Solution {
public:
    string removeKdigits(string num, int k) {
        stack<char> stk;
        for(auto x:num)
        {
            while(!stk.empty()&&stk.top()>x&&k)
            {
                stk.pop();
                k--;
            }
            stk.push(x);
        }
        while(k--) stk.pop();
        string res;
        while(stk.size())
        {
            res+=stk.top();
            stk.pop();
        }
        reverse(res.begin(),res.end());
        int i=0;
        while(i<res.size()&&res[i]=='0') i++;
        if(i==res.size()) return "0";
        return res.substr(i);
    }
};
```



## 链表

### 160、相交链表（双指针思想）

指针相遇意思就是：值和地址都相同

A和B两个链表长度可能不同，但是A+B和B+A的长度是相同的，所以遍历A+B和遍历B+A一定是同时结束。 如果A,B相交的话A和B有一段尾巴是相同的

```c++
/*
如：
[4,1,8,4,5]N,5,0,1,8,4,5
[5,0,1,8,4,5]N,4,1,8,4,5 
*/
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *p1=headA;
        ListNode *p2=headB;
        while(p1!=p2) //值和地址都一样才算相等
        {
            p1=(p1==NULL)?p1=headB:p1=p1->next;
            p2=(p2==NULL)?p2=headA:p2=p2->next;
            
        }
        return p1;
    }
};
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* a=headA;
        ListNode* b=headB;
        while(a!=b)
        {
            if(a==NULL)
            {
                a=headB;
            }
            else
                a=a->next;
            if(b==NULL)
            {
                b=headA;
            }
            else
                b=b->next;
             
        }
        return a;
    }
};
```

### 203、移除链表元素（添加虚头节点）

```
ListNode* removeElements(ListNode* head, int val) {
        
        ListNode *vir=new ListNode(0);
        vir->next=head;
        ListNode *cur=vir;
        while(cur->next!=NULL)
        {
            if(cur->next->val==val)
            {
                ListNode *tmp=cur->next;
                cur->next=cur->next->next;
                delete tmp;
            }
            else{
                cur=cur->next;
            }
        }
        return vir->next;
```

### 707、设计链表

```
class MyLinkedList {
public:
    // 定义链表节点结构体
    struct LinkedNode {
        int val;
        LinkedNode* next;
        LinkedNode(int val):val(val), next(nullptr){}
    };

    // 初始化链表
    MyLinkedList() {
        _dummyHead = new LinkedNode(0); // 这里定义的头结点 是一个虚拟头结点，而不是真正的链表头结点
        _size = 0;
    }

    // 获取到第index个节点数值，如果index是非法数值直接返回-1， 注意index是从0开始的，第0个节点就是头结点
    int get(int index) {
        if (index > (_size - 1) || index < 0) {
            return -1;
        }
        LinkedNode* cur = _dummyHead->next;
        while(index--){ // 如果--index 就会陷入死循环
            cur = cur->next;
        }
        return cur->val;
    }

    // 在链表最前面插入一个节点，插入完成后，新插入的节点为链表的新的头结点
    void addAtHead(int val) {
        LinkedNode* newNode = new LinkedNode(val);
        newNode->next = _dummyHead->next;
        _dummyHead->next = newNode;
        _size++;
    }

    // 在链表最后面添加一个节点
    void addAtTail(int val) {
        LinkedNode* newNode = new LinkedNode(val);
        LinkedNode* cur = _dummyHead;
        while(cur->next != nullptr){
            cur = cur->next;
        }
        cur->next = newNode;
        _size++;
    }

    // 在第index个节点之前插入一个新节点，例如index为0，那么新插入的节点为链表的新头节点。
    // 如果index 等于链表的长度，则说明是新插入的节点为链表的尾结点
    // 如果index大于链表的长度，则返回空
    void addAtIndex(int index, int val) {
        if (index > _size) {
            return;
        }
        LinkedNode* newNode = new LinkedNode(val);
        LinkedNode* cur = _dummyHead;
        while(index--) {
            cur = cur->next;
        }
        newNode->next = cur->next;
        cur->next = newNode;
        _size++;
    }

    // 删除第index个节点，如果index 大于等于链表的长度，直接return，注意index是从0开始的
    void deleteAtIndex(int index) {
        if (index >= _size || index < 0) {
            return;
        }
        LinkedNode* cur = _dummyHead;
        while(index--) {
            cur = cur ->next;
        }
        LinkedNode* tmp = cur->next;
        cur->next = cur->next->next;
        delete tmp;
        _size--;
    }

    // 打印链表
    
private:
    int _size;
    LinkedNode* _dummyHead;

};
```

### 206、反转链表(双指针)

```c++
//双指针法
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *tmp;
        ListNode *cur=head;
        ListNode *pre=NULL;
        while(cur)
        {
            tmp=cur->next;
            cur->next=pre;
            pre=cur;
            cur=tmp;
        }
        return pre;
    }
};
//递归法
class Solution {
public:
    ListNode* reverse(ListNode* pre,ListNode* cur){
        if(cur == NULL) return pre;
        ListNode* temp = cur->next;
        cur->next = pre;
        // 可以和双指针法的代码进行对比，如下递归的写法，其实就是做了这两步
        // pre = cur;
        // cur = temp;
        return reverse(cur,temp);
    }
    ListNode* reverseList(ListNode* head) {
        // 和双指针法初始化是一样的逻辑
        // ListNode* cur = head;
        // ListNode* pre = NULL;
        return reverse(NULL, head);
    }

};
```

### 92、反转链表II

```c++
//使用了递归
class Solution {
public:
    ListNode *lastNode;
    ListNode* reverseN(ListNode* head, int n)
    {
        if(n==1)
        {
            lastNode=head->next;
            return head;
        }
        ListNode *last=reverseN(head->next,n-1);
        head->next->next=head;
        head->next=lastNode;
        return last;
    }
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if(m==1)
        {
            return reverseN(head,n);
        }
        head->next=reverseBetween(head->next,m-1,n-1);
        return head;
    }
};
//解法2：
/*
   1 2 3 4 5 left=2,right=4
-->1 3 2 4 5
-->1 4 3 2 5
*/
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* dummy=new ListNode(-1);
        dummy->next=head;
        ListNode* pre=dummy;
        for(int i=0;i<left-1;i++)
        {
            pre=pre->next;
        }
        ListNode* cur=pre->next;
        for(int i=0;i<right-left;i++)
        {
            ListNode* next=cur->next;
            cur->next=next->next;
            next->next=pre->next;
            pre->next=next;
        }
        return dummy->next;
    }
};
//解法三
//依次找到需要反转的节点，原地反转
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        //特判
        if(left==right) return head;

        ListNode* dummy=new ListNode(-1);
        dummy->next=head;
        auto a=dummy;
        auto d=dummy;
        for(int i=0;i<left-1;i++) a=a->next;
        for(int i=0;i<right;i++) d=d->next;
        auto b=a->next,c=d->next;
        //反转b-d之间的部分
        for(auto p=b,q=b->next;q!=c;)
        {
            auto o=q->next;
            q->next=p;
            p=q;
            q=o;
        }
        
        b->next=c;a->next=d;
        return dummy->next;

    }
};
```

### 25、K个一组反转链表

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484597&idx=1&sn=c603f1752e33cb2701e371d84254aee2&chksm=9bd7fabdaca073abd512d8fff18016c9092ede45fed65c307852c65a2026d8568ee294563c78&scene=21#wechat_redirect

```c++

class Solution {
public:
    //反转[a,b)区间的链表
    ListNode *reverse(ListNode*a,ListNode* b)
    {
        ListNode* pre=NULL;
        ListNode*cur=a;
        ListNode *tmp;
        while(cur!=b)
        {
            tmp=cur->next;
            cur->next=pre;
            pre=cur;
            cur=tmp;
        }
        return pre;
    }
    //k个一组进行反转链表
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode*a=head;
        ListNode *b=head;
        for(int i=0;i<k;i++)
        {
            if(b==NULL) return head;  //不够K的情况
            b=b->next;
        }
        //反转前K个元素
        ListNode* newnode=reverse(a,b);
        a->next=reverseKGroup(b,k);
        return newnode;
    }
};
```



### 142、环形链表II(双指针)



![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4aFV2OYN4V4oCGSglPhyQMW5uN2KrTcjrnuQJ3qYERliaADMrXdZOr1959GWicQ0w7tTXwwX1bQwDQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

slow指针走过的节点数为: `x + y`， fast指针走过的节点数：`x + y + n (y + z)`，n为fast指针在环内走了n圈才遇到slow指针， （y+z）为 一圈内节点的个数A。

(x + y) * 2 = x + y + n (y + z)------->`x = n (y + z) - y` ,**从头结点出发一个指针，从相遇节点 也出发一个指针，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是 环形入口的节点**

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            // 快慢指针相遇，此时从head 和 相遇点，同时查找直至相遇
            if (slow == fast) {
                ListNode* index1 = fast;
                ListNode* index2 = head;
                while (index1 != index2) {
                    index1 = index1->next;
                    index2 = index2->next;
                }
                return index2; // 返回环的入口
            }
        }
        return NULL;
    }
};
```

### 21、合并两个有序链表

```c++
class Solution {
public:
	//递归
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==NULL)
        {
            return l2;
        }
        if(l2==NULL)
        {
            return l1;
        }
        if(l1->val<=l2->val)
        {
            l1->next=mergeTwoLists(l1->next,l2);
            return l1;
        }
        else
        {
            l2->next=mergeTwoLists(l1,l2->next);
            return l2;
        }
    }
};

class Solution {
public:
    //迭代
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode *dummy=new ListNode(-1);
        ListNode *res=dummy;
        while(l1&&l2)
        {
            if(l1->val<l2->val)
            {
                dummy->next=l1;
                l1=l1->next;
            }
            else
            {
                dummy->next=l2;
                l2=l2->next;
            }
            dummy=dummy->next;
        }
        dummy->next=(l1)?l1:l2;
        return res->next;

    }
};
```

### 23、合并K个升序链表（分治、优先队列）

```c++
//分治
class Solution {
public:
    // 合并两个有序链表
    ListNode* merge(ListNode* p1, ListNode* p2){
        if(!p1) return p2;
        if(!p2) return p1;
        if(p1->val <= p2->val){
            p1->next = merge(p1->next, p2);
            return p1;
        }else{
            p2->next = merge(p1, p2->next);
            return p2;
        }
    }
	//从中间一分为2
    ListNode* merge(vector<ListNode*>& lists, int start, int end){
        if(start == end) return lists[start];
        int mid = (start + end) / 2;
        ListNode* l1 = merge(lists, start, mid);
        ListNode* l2 = merge(lists, mid+1, end);
        return merge(l1, l2);
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if(lists.size() == 0) return nullptr;
        return merge(lists, 0, lists.size()-1);
    }
};
//优先队列：因为链表都是有序的
//一开始是把k个链表的首节点放到队列里。后面每次弹出一个节点，然后再判断这个节点有没有后继节点，有的话再放进队列中。
class Solution {
public:
    // 小根堆的回调函数
    //堆顶是小元素
    struct cmp{  
       bool operator()(ListNode *a,ListNode *b){
          return a->val > b->val;
       };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*, vector<ListNode*>, cmp> pri_queue;
        // 建立大小为k的小根堆
        for(auto elem : lists){
            if(elem) pri_queue.push(elem);
        }
        // 可以使用哑节点/哨兵节点
        ListNode dummy(-1);
        ListNode* p = &dummy;
        // 开始出队
        while(!pri_queue.empty()){
            ListNode* top = pri_queue.top(); 
			pri_queue.pop();
            p->next = top; p = top;
            if(top->next) pri_queue.push(top->next);
        }
        return dummy.next;  
    }
};
```

### 83、删除链表中的排序元素

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* cur = head;
        while (cur != nullptr && cur->next != nullptr) {
            if (cur->val == cur->next->val) {
                cur->next = cur->next->next;  //删除元素，并没有移动节点
            }
            else {
                cur = cur->next;
            }
        }
        return head;
    }
};
```

### 82、删除排序链表中的中的重复元素||---重复的都删除

```c++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(head==NULL) return head;
        //设置哑节点
        ListNode *dummy=new ListNode(-1);
        dummy->next=head;
        ListNode *cur=dummy;  //一直走的遍历节点
        while(cur->next&&cur->next->next)
        {
            if(cur->next->val==cur->next->next->val)
            {
                //一个个的删除
                int x=cur->next->val;
                while(cur->next&&cur->next->val==x)
                {
                    cur->next=cur->next->next;
                }
            }
            else
            {
                cur=cur->next;
            }
        }
        return dummy->next;
    }
};
```

### 剑指 18、删除链表的节点

```c++
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode *dummy=new ListNode(-1);
        dummy->next=head;
        ListNode *cur=dummy;
        while(cur->next)
        {
            if(cur->next->val==val)
            {
                cur->next=cur->next->next;
                break;
            }
            else
            {
                cur=cur->next;
            }
        }
        return dummy->next;
    }
};
```



### 哈希表

### 242、有效的字母异位词

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

```c++
//哈希表思路：把字符映射到数组也就是哈希表的索引下表上，再遍历字符串s的时候，「只需要将 s[i] - ‘a’ 所在的 //元素做+1 操作即可，同样在遍历字符串t的时候，对t中出现的字符映射哈希表索引上的数值再做-1的操作，
//如果record数组所有元素都为零0，说明字符串s和t是字母异位词
class Solution {
public:
    bool isAnagram(string s, string t) {
        int record[26] = {0};
        for (int i = 0; i < s.size(); i++) {
            // 并不需要记住字符a的ASCII，只要求出一个相对数值就可以了
            record[s[i] - 'a']++;
        }
        for (int i = 0; i < t.size(); i++) {
            record[t[i] - 'a']--;
        }
        for (int i = 0; i < 26; i++) {
            if (record[i] != 0) {
                // record数组如果有的元素不为零0，说明字符串s和t 一定是谁多了字符或者谁少了字符。
                return false;
            }
        }
        // record数组所有元素都为零0，说明字符串s和t是字母异位词
        return true;
    }
};
```

### 349、两个数组的交集

![img](https://mmbiz.qpic.cn/mmbiz_png/ciaqDnJprwv4NU6kiblLUBqKdRgiadjWxNkc3Dian1YUKtMBamhRv6VP12j0bAIKjuh47lHky8wlwTJHEl77QV0AicA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        unordered_set<int> result_set; // 存放结果
        unordered_set<int> nums_set(nums1.begin(), nums1.end());
        for (int num : nums2) {
            // 发现nums2的元素 在nums_set里又出现过
            if (nums_set.find(num) != nums_set.end()) {
                result_set.insert(num);
            }
        }
        return vector<int>(result_set.begin(), result_set.end());
    }
};
```

### 202、快乐数

输入：19
输出：true
解释：
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1

```
//set方法
class Solution {
public:
    // 取数值各个位上的单数之和
    int getSum(int n) {
        int sum = 0;
        while (n) {
            sum += (n % 10) * (n % 10);
            n /= 10;
        }
        return sum;
    }
    bool isHappy(int n) {
        unordered_set<int> set;
        while(1) {
            int sum = getSum(n);
            if (sum == 1) {
                return true;
            }
            // 如果这个sum曾经出现过，说明已经陷入了无限循环了，立刻return false
            if (set.find(sum) != set.end()) {
                return false;
            } else {
                set.insert(sum);
            }
            n = sum;
        }
    }
};
//快慢指针
class Solution {
public:
    int bitSquareSum(int n) {
        int sum = 0;
        while(n > 0)
        {
            int bit = n % 10;
            sum += bit * bit;
            n = n / 10;
        }
        return sum;
    }
    
    bool isHappy(int n) {
        int slow = n, fast = n;
        do{
            slow = bitSquareSum(slow);
            fast = bitSquareSum(fast);
            fast = bitSquareSum(fast);
        }while(slow != fast);   //如果最后相遇分两种情况：为1或者进入了环
        
        return slow == 1;
    }
};

```

### 454：四数相加II

和1、两数相加思路相同

给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。

```
class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> umap; //key:a+b的数值，value:a+b数值出现的次数
        // 遍历大A和大B数组，统计两个数组元素之和，和出现的次数，放到map中 
        for (int a : A) {
            for (int b : B) {
                umap[a + b]++;
            }
        }
        int count = 0; // 统计a+b+c+d = 0 出现的次数
        // 在遍历大C和大D数组，找到如果 0-(c+d) 在map中出现过的话，就把map中key对应的value也就是出现次数统计出来。
        for (int c : C) {
            for (int d : D) {
                if (umap.find(0 - (c + d)) != umap.end()) {
                    count += umap[0 - (c + d)];
                }
            }
        }
        return count;
    }
};

```

### 383、赎金信

给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。

和242题目差不多

```
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        int record[26] = {0};
        for (int i = 0; i < magazine.length(); i++) {
            // 通过recode数据记录 magazine里各个字符出现次数
            record[magazine[i]-'a'] ++; 
        }
        for (int j = 0; j < ransomNote.length(); j++) {
            // 遍历ransomNote，在record里对应的字符个数做--操作
            record[ransomNote[j]-'a']--; 
            // 如果小于零说明 magazine里出现的字符，ransomNote没有
            if(record[ransomNote[j]-'a'] < 0) {
                return false;
            }
        }
        return true;
    }
};
```

### 217、存在重复元素

```
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_map<int,int> map;
        if(nums.size()<=1)
            return false;
        for(int i=0;i<nums.size();i++)
        {
            map[nums[i]]++;
            if(map[nums[i]]>1)
                return true;
        }
        return false;
    }
};
```

### 287、寻找重复数（二分）

```c++
//抽屉原理
/*
n+1个苹果放进n个抽屉里，至少有一个抽屉会放两个苹果
划分之后，左右两个区间里至少存在一个区间，区间中数的个数大于区间长度
*/
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int left=1;  //注意从1开始
        int right=nums.size()-1;
        while(left<right)
        {
            //mid不是代表下标而是中间的那个数
            int mid = left + right >> 1;
            int count=0;
            for(auto num:nums)
            {
                if(num<=mid)
                    count++;
            }
            if(count>mid)
                right=mid;
            else
                left=mid+1;
        }
        return left;
    }
};
```



### 219、存在重复元素II

```
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int,int> map;
        if(nums.size()<=1)
            return false;
        for(int i=0;i<nums.size();i++)
        {
            map[nums[i]]++;
            if(map[nums[i]]>1)
                return true;
            if(map.size()>k)
                map.erase(nums[i-k]);
        }
        return false;
    }
};
```

### 220、存在重复元素III

```c++
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {  
        std::set<long> s;
        for (int i = 0; i < nums.size(); ++i) {
            auto pos = s.lower_bound(long(nums[i])-t);
            ////< @attention
            //set的内置函数lower_bound可以找到第一个大于等于nums[i]-t的数
            if (pos!=s.end() && *pos<=long(nums[i])+t) {return true;}
            s.insert(nums[i]);
            if (s.size() > k) {s.erase(nums[i-k]);} ////< @note 维护活动窗口
        }
        return false;
    }
};
```

### 剑指 56、数组中数字出现的次数II

```c++
//哈希表
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int,int> map;
        for(auto i:nums)
        {
            map[i]++;
        }
        for(auto [k,v]:map)
        {
            if(v==1)
                return k;
        }
        return 0;
         
    }
};
//排序
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size()-2;i+=3)  //每三个数检查一次
        {
            if(nums[i]!=nums[i+2])
                return nums[i];
        }
        return nums.back();
    }
};
```

### 61、旋转链表

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        //k如果很大是没有意义的，对链表长度取模
        if(!head) return NULL;
        int n=0;
        auto cur=head;
        while(cur)
        {
            n++;
            cur=cur->next;
        }
        k%=n;
        //k--;
        auto fast=head;
        auto slow=head;
        while(k--)
        {
            fast=fast->next;
        }
        while(fast->next)
        {
            slow=slow->next;
            fast=fast->next;
        }
        fast->next=head;
        auto node=slow->next;
        slow->next=NULL;
        return node;

    }
};
```

### 143、重排链表

```c++
//思路：先找到后半段，反转后半段，再依次插入
class Solution {
public:
    void reorderList(ListNode* head) {
        //特判
        if(!head||!head->next||!head->next->next) return;
        ListNode* slow = head;
        ListNode* quick = head;
        //找到中间节点（如果有两个中间节点,slow是靠后的那个）
        while(quick && quick->next)
        {
            slow = slow->next;
            quick = quick->next->next;
        }
        ListNode* pre = slow;
        ListNode* cur = slow->next;
        slow->next = NULL;
        //之所以让后一段的尾节点指向第一段的尾节点是因为我们要用这个做判断什么时候插入结束
        //反转slow后的链表
        while(cur)
        {
            ListNode* temp = cur->next;
            cur->next = pre;
            pre = cur;
            if(temp == NULL)
                break;
            cur = temp;
        }
        while(head!=cur&&head->next!=cur)
        {
            ListNode* tmp=cur->next;
            cur->next=head->next;
            head->next=cur;
            cur=tmp;
            head=head->next->next;
        }
    }
};
//解法二
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    void reorderList(ListNode* head) {
        auto slow = head, fast = head;
        //finding middle point
        //利用快慢指针找到中心点
        while(fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        //reversing the last half
        //将后半部分反转
        auto tail = reverseList(slow->next);
        slow->next = NULL;

        //inseting the last half to the first half
        //将后半部分插入前半段
        auto headptr = head, tailptr = tail;
        while(tailptr)
        {
            auto temp1 = headptr->next;
            auto temp2 = tailptr->next;
            headptr->next = tailptr;
            tailptr->next = temp1;
            headptr = temp1;
            tailptr = temp2;
        }
    }
	//反转链表
    ListNode* reverseList(ListNode* head)
    {
        if(!head || !head->next)
            return head;
        ListNode* thisNode = head;
        ListNode* last = NULL;
        while(thisNode)
        {
            ListNode* temp = thisNode->next;
            thisNode->next = last;
            last = thisNode;
            thisNode = temp;
        }
        return last;
    }
};
```

### 328、奇偶链表

```c++
/*
从前往后遍历整个链表，遍历时维护四个指针：奇数链表头结点，奇数链表尾节点，偶数链表头结点，偶数链表尾节点。
遍历时将位置编号是奇数的节点插在奇数链表尾节点后面，将位置编号是偶数的节点插在偶数链表尾节点后面。
遍历完整个链表后，将偶数链表头结点插在奇数链表尾节点后面即可
*/
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        //特判
        if(!head||!head->next) return head;
        //定义指针
        ListNode* odd=head;
        ListNode* even=head->next;
        ListNode* even_head=head->next;
        //下面就该移动指针，连接指针
        for(ListNode* p=head->next->next;p;)
        {
            odd->next=p;
            odd=p;
            p=p->next;
            if(p)
            {
                even->next=p;
                even=p;
                p=p->next;
            }
        }
        //拼接
        odd->next=even_head;
        even->next=0;
        return head;
    }
};
```

### 138、复制带随机指针的链表（哈希+递归）

```c++
//递归
/*
哈希表Mydic映射原有节点->新的节点
原节点为空，则返回空
原节点在哈希表中可以找到，则说明新的节点已生成，直接返回
根据原有节点的值，创建新的节点root = Node(node.val)
将原有节点和新节点的对应关系添加到哈希表中Mydic[node] = root
最后参照原节点的next和random关系，创建新的next和random节点给新节点root
递归整个过程
*/
class Solution {
public:
    unordered_map<Node*, Node*> visited;
    Node* copyRandomList(Node* head) {
        if(!head) return head;

        if(visited.find(head) != visited.end())
            return visited[head];
        Node* cloneNode = new Node(head->val);
        visited[head] = cloneNode;

        cloneNode->next = copyRandomList(head->next);
        cloneNode->random = copyRandomList(head->random);
        return cloneNode;
    }
};
//非递归
class Solution {
public:
    Node* copyRandomList(Node* head) {
      if( head == NULL ){
            return head;
        }
        unordered_map< Node*,Node* > map;
        Node* newhead = new Node( head->val );
        map[head] = newhead;

        auto pre = newhead;
        for(auto it = head->next ; it!=NULL ; it = it->next ){
            Node* cur = new Node( it->val );
            map[it] = cur;
            pre->next = cur;
            pre = cur;
        }
        for(auto it = head ; it!=NULL ; it = it->next ){
            map[it]->random = map[it->random];
        }
        return newhead;
    }
};
```

### 876、链表的中间节点(快慢指针)

```c++
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* slow=head;
        ListNode* fast=head;
        while(fast&&fast->next)
        {
            slow=slow->next;
            fast=fast->next->next;
        }
        return slow;
    }
};
```

### 86、分割链表

```c++
//新建两个链表，分别存储小于x和大于等于x，最后将两个链表合并
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* minl=new ListNode(0);
        ListNode* cur1=minl;

        ListNode* maxl=new ListNode(0);
        ListNode* cur2=maxl;

        ListNode* cur=head;
        while(cur!=nullptr)
        {
            if(cur->val<x)
            {
                cur1->next=cur;
                cur1=cur1->next;
            }
            else
            {
                cur2->next=cur;
                cur2=cur2->next;
            }
            cur=cur->next;
        }
        cur2->next=NULL;
        cur1->next=maxl->next;
        return minl->next;
    }
};
```



## 位运算

### 461、汉明距离

这两个数字对应二进制位不同的位置的数目

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int n=x^y;
        int count=0;
        while(n)
        {
            count++;
            n=n&(n-1);  //结果总能消除n的末位的1
        }
	    return count;
    }
};


class Solution {
public:
    int hammingDistance(int x, int y) {
       
       return bitset<32>(x^y).count();
    }
};
```

### 191、位1的个数

```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count=0;
        int i=32;
        int t=1;
        while (i -- )               //32位循环32次
        {
            if (n & 1) count ++ ;     //判断最低位是否为1
            n >>= 1;                //n每次向右移动一位，即除以2
        }
        return count;
    }
};
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count=0;
        while(n)
        {
            n&=(n-1);  //n的二进制位中的最低位的 1变为 0
            count++;
        }
        return count;
    }
};
```



### 136、只出现一次的数字（异或）

利用异或运算，异或运算的性质：

​    任何数和 0做异或运算，结果仍然是原来的数，即 a⊕0=a
​    任何数和其自身做异或运算，结果是 000，即 a⊕a=0
​    异或运算满足交换律和结合律，即 a⊕b⊕a=b⊕a⊕a=b⊕(a⊕a)=b⊕0=b



```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ret=0;
        for(int i=0;i<nums.size();i++)
        {
            ret=ret^nums[i];
        }
        return ret;
    }
};
```

### 268、丢失的数字

输入：nums = [3,0,1]
输出：2
解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。

```c++
法一：利用异或
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int res = nums.size();
    	for(int i = 0; i < nums.size(); ++i)
        	res = res ^ i ^ nums[i];            // a^b^b = a;
    	return res ;        
    }
};
法二：高斯公式
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int res = nums.size();
    	int result=(res+1)*res/2;
        for(auto e:nums)
            result-=e;
    	return result ;        
    }
};
```

### 260、只出现一次的数字

给定一个整数数组 `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

```c++
//排序解决
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        vector<int> v;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size()-1;i++)
        {
            if(nums[i+1]-nums[i]!=0)
            {
                v.push_back(nums[i]);
            }
            else 
            {
                i=i+1;
            }
        }
        if(v.size()==1)
            v.push_back(nums.back());
        return v;
    }
};
//位运算：通过低位来区分两个数；
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int sign = 0;
    //取得数组中两个唯一数的按位异或结果
    for (int i = 0; i < nums.size(); i++)
    {
        sign ^= nums[i];
    }
    //获取区分两个唯一数的比特位所代表的值
    //n&(-n) 得到 n 的位级表示中最低的那一位 1
    //也可以写成：sign &= (~sign) + 1
    sign &= -sign;
    int n1 = 0, n2 = 0;
    //通过标识，区分两个数组
    for (int i = 0; i < nums.size(); i++)
    {
        if ((nums[i] & sign) == sign)   //注意括号的优先级
            n1 ^= nums[i];
        else
            n2 ^= nums[i]; 
    }
    return { n1,n2 };

    }
};
```

### 371、两整数之和（不使用+-*/）

```
class Solution {
public:
    int getSum(int a, int b) {
        int sum=0;
        int arr=0;
        do
        {
            sum=a^b;
            arr=unsigned(a&b)<<1;  //当a & b的结果是负数时，左移就会造成符号位的溢出
            a=sum;
            b=arr;
        }while(b!=0);
        return a;
    }
};
```



## 树

**把题目的要求细化，搞清楚根节点应该做什么，然后剩下的事情抛给前/中/后序的遍历框架就行了**

递归三部曲：

**1、确定递归函数的参数和返回值**

2、**确定终止条件**

**3、确定单层递归的逻辑**

### 104、二叉树的最大深度

```c++
//后序遍历：依然是因为要通过递归函数的返回值做计算树的高度。
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;

    }

};
```

### 173、二叉搜索树迭代器

```c++
//就相当于把用栈遍历树的步骤拆开了
class BSTIterator {
public:
    stack<TreeNode*> stk;
    BSTIterator(TreeNode* root) {
        while(root)
        {
            stk.push(root);
            root=root->left;
        }
    }
    
    int next() {
        auto p=stk.top();
        stk.pop();
        auto res=p->val;
        p=p->right;
        while(p)
        {
            stk.push(p);
            p=p->left;
        }
        return res;
    }
    
    bool hasNext() {
        return !stk.empty();
    }
};
```

### 297、二叉树的序列化和反序列化

![image-20210811142451256](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20210811142451256.png)

```c++

class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string res;
        dfs1(root,res);
        return res;
    }
    void dfs1(TreeNode* root,string &res)
    {
        //前序遍历
        if(!root)
        {
            res+="#,";
            return ;
        }
        res+=to_string(root->val)+',';
        dfs1(root->left,res);
        dfs1(root->right,res);

    }
    // Decodes your encoded data to tree.
    //反序列化的时候，给你的是一个字符串，你要判断负数的这种情况
    TreeNode* deserialize(string data) {
        int u=0;
        return dfs2(data,u);
    }
    TreeNode* dfs2(string& data,int &u)
    {
        if(data[u]=='#')
        {
            u+=2;
            return NULL;
        }  
        //求出当前节点的值，可能是负数
        int t=0;
        bool is_minus=false;
        if(data[u]=='-')
        {
            is_minus=true;
            u++;
        } 
        while(data[u]!=',')
        {
            t=t*10+data[u]-'0';
            u++;
        }
        u++;
        if(is_minus) t=-t;
        auto root=new TreeNode(t);
        root->left=dfs2(data,u);
        root->right=dfs2(data,u);
        return root;
    }
};

```

### 129、求根节点到叶子节点点数之和

```c++
class Solution {
public:
    void dfs(TreeNode* root,int res,int &sum)
    {
        if(!root) return;
        res=res*10+root->val;
        if(root->left==NULL&&root->right==NULL)
        {
            //表明是叶子节点了
            sum+=res;
            return;
        }
        dfs(root->left,res,sum);
        dfs(root->right,res,sum);
    }
    int sumNumbers(TreeNode* root) {
        int sum=0;
        int res=0;
        dfs(root,0,sum);
        return sum;
    }
};
```



### 110、判断是否是平衡二叉树

```c++
int maxdepth(TreeNode* root);
class Solution {
private:
    bool flag = 1;
public:
    bool isBalanced(TreeNode* root) {

            maxdepth(root);
            return flag;
    }
    int maxdepth(TreeNode* root)
    {
        if(root==nullptr)
            return 0;
        int l=maxdepth(root->left);
        int r=maxdepth(root->right);
        if(abs(l-r)>1)
            flag=0;
        
        return max(l,r)+1;
    }

};
```

### 543、二叉树的直径

两结点之间的路径长度是以它们之间边的数目表示。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int max_len=0;
    int dfs(TreeNode* root)
    {
        if(root==nullptr)
            return 0;
        int l=dfs(root->left);
        int r=dfs(root->right);
        max_len=max(max_len,l+r+1);
        return max(l,r)+1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return max_len-1;   //深度=节点数-1
    }
};
```

### 226、翻转二叉树

```c++
方法一：交换前先把右边的树交换好
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        
        if(root==NULL)
            return NULL;
        TreeNode* root_left=root->left;
        root->left=invertTree(root->right);
        root->right=invertTree(root_left);
        return root;

        
    }
};
方法二：先交换左右子树，再交换左右子树的子树
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(root==NULL)
            return NULL;
        TreeNode* temp=root->left;
        root->left=root->right;
        root->right=temp;
        invertTree(root->left);//递归左子树
        invertTree(root->right);//递归右子树 
        return root;
    }
};

```

### 剑指offer 27、二叉树的镜像

```c++
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        // 考虑为空情况
        if (root == nullptr)
        {
            return nullptr;
        }
        // left 和 right 反序, 临时保存left结点
        TreeNode* temp = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(temp); 
        return root;
    }
};
```



### 617、合并二叉树

```
法一：递归
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        
    if (t1 == NULL) return t2;
    if (t2 == NULL) return t1;

        t1->val+=t2->val;   //确定单层递归的逻辑
        t1->left=mergeTrees(t1->left,t2->left);
        t1->right=mergeTrees(t1->right,t2->right);
        return t1;
    }
};
```

### 144、二叉树的前序遍历

```c++
//法一：递归遍历
class Solution {
public:
    void qian_bianli(TreeNode* root,vector<int>& vec)
    {
        if(root==nullptr)
            return;
        vec.push_back(root->val);
        qian_bianli(root->left,vec);
        qian_bianli(root->right,vec);
    }
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> v;
        qian_bianli(root,v);
        return v;
    }


};
//法二：迭代法
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> result;
        st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();                      // 中
            st.pop();
            if (node != NULL) result.push_back(node->val);
            else continue;
            st.push(node->right);                           // 右
            st.push(node->left);                            // 左
        }
        return result;
    }
};
```

### 94、二叉树的中序遍历

```c++
//法一：递归遍历
class Solution {
public:
    void qian_bianli(TreeNode* root,vector<int>& vec)
    {
        if(root==nullptr)
            return;
        qian_bianli(root->left,vec);
        vec.push_back(root->val);
        
        qian_bianli(root->right,vec);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> v;
        qian_bianli(root,v);
        return v;
    }


};
//法二：迭代法
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> st;
        TreeNode* cur = root;
        while (cur != NULL || !st.empty()) {
            if (cur != NULL) { // 指针来访问节点，访问到最底层
                st.push(cur); // 讲访问的节点放进栈
                cur = cur->left;                // 左
            } else {
                cur = st.top(); // 从栈里弹出的数据，就是要处理的数据（放进result数组里的数据）
                st.pop();
                result.push_back(cur->val);     // 中
                cur = cur->right;               // 右
            }
        }
        return result;
    }
};
//迭代写法二
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        TreeNode* p=root;
        while(p||st.size())
        {
            while(p)
            {
                st.push(p);
                p=p->left;
            }
            p=st.top();
            st.pop();
            res.push_back(p->val);
            p=p->right;
        }

        return res;
    }
};
```

### 145、二叉树的后序遍历

```c++
//法一：递归遍历
class Solution {
public:
    void qian_bianli(TreeNode* root,vector<int>& vec)
    {
        if(root==nullptr)
            return;
        qian_bianli(root->left,vec);
        qian_bianli(root->right,vec);
        vec.push_back(root->val);
        
        
    }
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> v;
        qian_bianli(root,v);
        return v;
    }


};
//法二：迭代
class Solution {
public:

    vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> result;
        st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            st.pop();
            if (node != NULL) result.push_back(node->val);
            else continue;
            st.push(node->left); // 相对于前序遍历，这更改一下入栈顺序
            st.push(node->right);
        }
        reverse(result.begin(), result.end()); // 将结果反转之后就是左右中的顺序了
        return result;
    }
};
//雪菜大神  迭代
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        auto p=root;
        while(p||st.size())
        {
            while(p)
            {
                res.push_back(p->val);
                st.push(p);
                p=p->right;
            }
            p=st.top();
            st.pop();
            p=p->left;
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```

### 102、二叉树的层序遍历

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        queue<TreeNode*>q;
        if(root!=NULL)
            q.push(root);
        while(!q.empty())
        {
            vector<int> v;
            int size=q.size();
            for(int i=0;i<size;i++)  //用来表示每一层有多少个数
            {
                TreeNode* node = q.front();
                q.pop();
                v.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
             result.push_back(v);
        }
            return result;
    }
};
```

### 103、二叉树的锯齿形层序遍历

```c++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        queue<TreeNode *> que;
        if(root!=nullptr)
            que.push(root);
        int flag=false;
        while(!que.empty())
        {
            int size=que.size();
            vector<int> path;
            for(int i=0;i<size;i++)
            {
                auto it=que.front();
                que.pop();
                path.push_back(it->val);
                if(it->left) que.push(it->left);
                if(it->right) que.push(it->right);
            }
            if(flag==false)
                res.push_back(path);
            else
            {
                reverse(path.begin(),path.end());
                res.push_back(path);
            }
            flag=!flag;

        }
        return res;
    }
};
```



### 107、二叉树的层序遍历II

给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

```
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        queue<TreeNode*> que;
        if (root != NULL) que.push(root);
        vector<vector<int>> result;
        while (!que.empty()) {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; i++) { 
                TreeNode* node = que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            result.push_back(vec);
        }
        reverse(result.begin(), result.end()); // 在这里反转一下数组即可
        return result;

    }
};
```

### 199、二叉树的右视图

```c++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*>q;
        vector<int> v;
        if(root!=NULL)
            q.push(root);
        while(!q.empty())
        {
            
            int size=q.size();
            for(int i=0;i<size;i++)  //用来表示每一层有多少个数
            {
                TreeNode* node = q.front();
                q.pop();
                if(i==size-1)
                    v.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            
        }
         return v;
    }
};

```

### 637、二叉树的层平均值

```
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode*>q;
        vector<double> v;
        double sum=0;
        if(root!=NULL)
            q.push(root);
        while(!q.empty())
        { 
            int size=q.size();
            for(int i=0;i<size;i++)  //用来表示每一层有多少个数
            {
                TreeNode* node = q.front();
                q.pop();
                sum+=node->val;   //把每一层的数据进行相加
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
             v.push_back(sum/size);
             sum=0;
        }
        return v;
    }
};
```

### 429、N叉树的层序遍历

```
class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        queue<Node*> que;
        if (root != NULL) que.push(root);
        vector<vector<int>> result;
        while (!que.empty()) {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; i++) { 
                Node* node = que.front();
                que.pop();
                vec.push_back(node->val);
                for (int i = 0; i < node->children.size(); i++) { // 将节点孩子加入队列
                    if (node->children[i]) que.push(node->children[i]);
                }
            }
            result.push_back(vec);
        }
        return result;

    }
};
```

### 101、对称二叉树

```c++
//递归
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        return dfs(root->left,root->right);
    }
    bool dfs(TreeNode* p,TreeNode* q)
    {
        if(!p||!q) return !p&&!q;
        return p->val==q->val&&dfs(p->left,q->right)&&dfs(p->right,q->left);
    }
};
//迭代
//左子树：左 中 右
//右子树：右 中 左
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        stack<TreeNode*> left,right;
        auto l=root->left,r=root->right;
        while(l||r||left.size()||right.size())
        {
            while(l&&r)
            {
                left.push(l);
                l=l->left;
                right.push(r);
                r=r->right;
            }
            if(l||r) return false;
            l=left.top();
            left.pop();
            r=right.top();
            right.pop();
            if(l->val!=r->val) return false;
            l=l->right;
            r=r->left;
        }
        return true;
    }
};

//递归
class Solution {
public:
    bool compare(TreeNode* left, TreeNode* right) {
        // 首先排除空节点的情况
        if (left == NULL && right != NULL) return false;
        else if (left != NULL && right == NULL) return false;
        else if (left == NULL && right == NULL) return true;
        // 排除了空节点，再排除数值不相同的情况
        else if (left->val != right->val) return false;

        // 此时就是：左右节点都不为空，且数值相同的情况
        // 此时才做递归，做下一层的判断
        bool outside = compare(left->left, right->right);   // 左子树：左、 右子树：右
        bool inside = compare(left->right, right->left);    // 左子树：右、 右子树：左
        bool isSame = outside && inside;                 // 左子树：中、 右子树：中 （逻辑处理）
        return isSame;

    }
    bool isSymmetric(TreeNode* root) {
        if (root == NULL) return true;
        return compare(root->left, root->right);
    }
};

//迭代
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if (root == NULL) return true;
        queue<TreeNode*> que;
        que.push(root->left);   // 将左子树头结点加入队列
        que.push(root->right);  // 将右子树头结点加入队列
        while (!que.empty()) {  // 接下来就要判断这这两个树是否相互翻转
            TreeNode* leftNode = que.front(); que.pop();    
            TreeNode* rightNode = que.front(); que.pop();
            if (!leftNode && !rightNode) {  // 左节点为空、右节点为空，此时说明是对称的
                continue;
            }

            // 左右一个节点不为空，或者都不为空但数值不相同，返回false
            if ((!leftNode || !rightNode || (leftNode->val != rightNode->val))) { 
                return false;
            }
            que.push(leftNode->left);   // 加入左节点左孩子
            que.push(rightNode->right); // 加入右节点右孩子
            que.push(leftNode->right);  // 加入左节点右孩子
            que.push(rightNode->left);  // 加入右节点左孩子
        }
        return true;
    }
};


```

### 559、N叉树的最大深度

```
//迭代（队列）
class Solution {
public:
    int maxDepth(Node* root) {
        int m=0;
        queue<Node*> que;
        if (root != NULL) que.push(root);        
        while (!que.empty()) {
            int size = que.size();
            m++;
            for (int i = 0; i < size; i++) { 
                Node* node = que.front();
                que.pop();
                
                for (int i = 0; i < node->children.size(); i++) { // 将节点孩子加入队列
                    if (node->children[i]) que.push(node->children[i]);
                }
            }
            
        }
        return m;
    }
};
//递归
class Solution {
public:
    int maxDepth(Node* root) {
        if (root == 0) return 0;
        int depth = 0;
        for (int i = 0; i < root->children.size(); i++) {
            depth = max (depth, maxDepth(root->children[i]));
        }
        return depth + 1;
    }
};
```

### 111、二叉树的最小深度

```
class Solution {
public:
    int getDepth(TreeNode* node) {
        if (node == NULL) return 0;
        int leftDepth = getDepth(node->left);    // 左
        int rightDepth = getDepth(node->right);  // 右
                                                 // 中
        // 当一个左子树为空，右不为空，这时并不是最低点
        if (node->left == NULL && node->right != NULL) { 
            return 1 + rightDepth;
        }   
        // 当一个右子树为空，左不为空，这时并不是最低点
        if (node->left != NULL && node->right == NULL) { 
            return 1 + leftDepth;
        }
        int result = 1 + min(leftDepth, rightDepth); 
        return result;
    }

    int minDepth(TreeNode* root) {
        return getDepth(root);
    }
};
//迭代
class Solution {
public:

    int minDepth(TreeNode* root) {
        if (root == NULL) return 0;
        int depth = 0;
        queue<TreeNode*> que;
        que.push(root);
        while(!que.empty()) {
            int size = que.size(); 
            depth++; // 记录最小深度
            int flag = 0;
            for (int i = 0; i < size; i++) {
                TreeNode* node = que.front();
                que.pop();
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
                if (!node->left && !node->right) { // 当左右孩子都为空的时候，说明是最低点的一层了，退出
                    flag = 1;
                    break;
                }
            }
            if (flag == 1) break;
        }
        return depth;
    }
};
```

### 662、二叉树最大宽度

```c++
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        //层序遍历
        queue<pair<TreeNode*,int>> que;  //pair的第二个位置记录当前是第几个节点
        if(!root) return 0;
        que.push({root,1});
        int max_wid=0;
        while(!que.empty())
        {
            int n=que.size();
            int start=que.front().second;  //start是本层起点
            int index;  //index是本层当前遍历到的节点的索引
            while(n--)
            {
                auto p=que.front().first;
                index=que.front().second;
                que.pop();
                if(p->left) que.push({p->left,index*2-start*2});  //防止索引位置太大溢出
                if(p->right) que.push({p->right,index*2+1-start*2});
            }
            max_wid=max(max_wid,index-start+1);
        }
        return max_wid;
        
    }
};
```

### 124、二叉树中的最大路径和

```c++
/*
	a
  b	  c
 有三条路径：a+b+c、a+b、a+c
 注意：最大值和获取一个结点返回的最大值不同
*/
class Solution {
public:
    int res=INT_MIN;
    int maxGin(TreeNode* root)
    {
        if(root==nullptr) return 0;
        int left=max(0,maxGin(root->left));
        int right=max(0,maxGin(root->right));
        int lmr=root->val+left+right;  //a+b+c
        int ret=root->val+max(left,right);
        res=max(res,max(lmr,ret));
        return ret;
    }
    int maxPathSum(TreeNode* root) {
        maxGin(root);
        return res;
    }
};
```



### 112、路径总和

```c++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root==NULL) return false;
        if(root->left==NULL&&root->right==NULL)
            return root->val==targetSum;
        int new_sum=targetSum-root->val;
        return hasPathSum(root->left,new_sum)||hasPathSum(root->right,new_sum);
    }
};
```

### 113、路径总和II

```c++
//思路：用递归方法做
//	  递归左右子树，newsum=targetSum-root->val
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<int> cur;
        vector<vector<int>> res;
        pathSum(root,targetSum,cur,res);
        return res;
    }
    void pathSum(TreeNode* root,int targetSum,vector<int>& cur,vector<vector<int>>& res)
    {
        if(root==NULL) return;
        if(root->left==NULL&&root->right==NULL)
        {
            if(root->val==targetSum)
            {
                cur.push_back(root->val);
                res.push_back(cur);
                cur.pop_back();
            }
            return;
        }
        int new_sum=targetSum-root->val;
        cur.push_back(root->val);
        pathSum(root->left,new_sum,cur,res);
        pathSum(root->right,new_sum,cur,res);
        cur.pop_back();
    }
};
```



### 222、完全二叉树的节点数

```c++
//迭代
class Solution {
public:
    int countNodes(TreeNode* root) {
        int m=0;
        queue<TreeNode*>q;
        if(root!=NULL)
            q.push(root);
        while(!q.empty())
        {
            vector<int> v;
            int size=q.size();
            for(int i=0;i<size;i++)  //用来表示每一层有多少个数
            {
                TreeNode* node = q.front();
                q.pop();
                m++;
                
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
             
        }
            return m;
    }
};
//递归
class Solution {
public:
    int countNodes(TreeNode* root) {
          return  getsum(root);       
    }
    int getsum(TreeNode* root)
    {
        if(root==NULL)
        {
            return 0;
        }
        int left=getsum(root->left);
        int right=getsum(root->right);
        return left+right+1;
    }
};
```

### 116、填充每个节点的下一个右侧节点指针

![image-20201027153422573](C:\Users\32184\AppData\Roaming\Typora\typora-user-images\image-20201027153422573.png)

```
****递归的本质就是每一个节点需要做什么
//法一：递归
class Solution {
private:
    void traversal(Node* cur) {
        if (cur == NULL) return;
                                // 中
        if (cur->left) 
        	cur->left->next = cur->right; // 操作1
        if (cur->right) 
        {
            if (cur->next) 
            	cur->right->next = cur->next->left; // 操作2 
            else 
            	cur->right->next = NULL;
        }
        traversal(cur->left);   // 左
        traversal(cur->right);  // 右
    }
public:
    Node* connect(Node* root) {
        traversal(root);
        return root;
    }
};

//法二：递归
Node connect(Node root) {
    if (root == null) return null;
    connectTwoNode(root.left, root.right);
    return root;
}

// 定义：输入两个节点，将它俩连接起来
void connectTwoNode(Node node1, Node node2) {
    if (node1 == null || node2 == null) {
        return;
    }
    /**** 前序遍历位置 ****/
    // 将传入的两个节点连接
    node1.next = node2;

    // 连接相同父节点的两个子节点
    connectTwoNode(node1.left, node1.right);
    connectTwoNode(node2.left, node2.right);
    // 连接跨越父节点的两个子节点
    connectTwoNode(node1.right, node2.left);
}
```

### 114、二叉树展开为链表

```c++
//递归，明确递归函数的作用
class Solution {
public:
    void flatten(TreeNode* root) {
        if(root==NULL)
            return;
        flatten(root->left);
        flatten(root->right);
        TreeNode* right_node=root->right;
        root->right=root->left;
        root->left=NULL;
        
        
        TreeNode* p=root;
        while(p->right!=NULL)
        {
            p=p->right;
        }
        
        p->right=right_node;
    }
};
```

### 654、最大二叉树

给定一个不含重复元素的整数数组。一个以此数组构建的最大二叉树定义如下：


	二叉树的根是数组中的最大元素。
	左子树是通过数组中最大值左边部分构造出的最大二叉树。
	右子树是通过数组中最大值右边部分构造出的最大二叉树。

```java
class Solution {
public:
    //根据索引构造树
    TreeNode* bulid(vector<int>& n,int lo,int hi)
    {
        if(lo>hi)
            return NULL;
        //找出了最大值和对应的索引
        int index=-1;
        int maxval=INT_MIN;
        for(int i=lo;i<=hi;i++)
        {
            if(n[i]>maxval)
            {
                maxval=n[i];
                index=i;
            }
        }

        TreeNode *node=new TreeNode(maxval);
        node->left=bulid( n,lo,index-1);
        node->right=bulid( n,index+1,hi);
        return node;

    }
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return bulid(nums,0,nums.size()-1);
    }
};
```

### 105、根据前序和中序构造二叉树

```c++

class Solution {
public:
    TreeNode * build(vector<int>& preorder,int prestart,int preend,vector<int>& inorder,int instart,int inend)
    {
        if(prestart>preend)
            return NULL;
        int val=preorder[prestart];
       

        int index=0;
        for(int i=instart;i<=inend;i++)
        {
            if(inorder[i]==val)
            {
                index=i;   //找出根节点在inorder的索引
                break;
            }
        } 
        TreeNode *node=new TreeNode(val);  //确定了根节点
        int size=index-instart;
        node->left=build(preorder,prestart+1,prestart+size,inorder,instart,index-1);
        node->right=build(preorder,prestart+size+1,preend,inorder,index+1,inend);

        return node;

    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return build(preorder,0,preorder.size()-1,inorder,0,inorder.size()-1);
    }
};
```

### 106、根据中序和后序构造二叉树

```

class Solution {
public:
    TreeNode* bulid(vector<int>& inorder, int prestart,int preend,vector<int>& postorder,int instart,int inend)
    {
        //终止条件
        if(prestart>preend)
        {
            return NULL;
        }
        int val=postorder[inend];
        int index=0;
        for(int i=prestart;i<=preend;i++)
        {
            if(inorder[i]==val)
            {
                index=i;
                break;
            }
        }
        int size=index-prestart;
        TreeNode *node=new TreeNode(val);
        node->left=bulid(inorder,prestart,index-1, postorder,instart,instart+size-1);
        node->right=bulid(inorder,index+1,preend, postorder,instart+size,inend-1);

        return node;

    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        return bulid(inorder,0,inorder.size()-1,postorder,0,postorder.size()-1);
    }
};






```

### 652、寻找重复的子树

```
class Solution {
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        unordered_map<string, int> counts;
        vector<TreeNode*> ans;
        serialize(root, counts, ans);
        return ans;
    }
private:
	//函数的定义：返回每一个节点的序列化（每一个节点当作根节点）
	//在这个函数内部：把序列化存入到unorderd_map中，如果哪一个次数超过1就把此节点放入容器中
    string serialize(TreeNode* root, unordered_map<string, int>& counts, vector<TreeNode*>& ans) {
        if (!root) return "#";
        string left=serialize(root->left, counts, ans);
        string right=serialize(root->right, counts, ans);
        string key = to_string(root->val) + "," + left + ","  + right;
        counts[key]++;
        if (counts[key] == 2)
            ans.push_back(root);
        return key;
    }
};
```

### 404、左叶子之和

```
class Solution {
public:
    //二叉树的所有左叶子之和
    int sumOfLeftLeaves(TreeNode* root) {
        if(root==NULL) return 0;
        int leftval=sumOfLeftLeaves(root->left);
        int rightval=sumOfLeftLeaves(root->right);
        int val=0;
        if(root->left!=NULL&&root->left->left==NULL&&root->left->right==NULL)
        {
            val=root->left->val;
        }
        return val+leftval+rightval;
    }
};
```

### 513、找树左下角的值

```
//层序遍历
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*>q;
        int result=0;
        if(root!=NULL)
            q.push(root);
        while(!q.empty())
        {
            
            int size=q.size();
            for(int i=0;i<size;i++)  //用来表示每一层有多少个数
            {
                TreeNode* node = q.front();
                q.pop();
                if(i==0) result=node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
             
        }
        return result;
    }
};
```



## 二叉搜索树：中序就是从小到大的顺序

### 108、将有序数组转换成二叉搜索树

```c++
//每次以中点为根，以左半部分为左子树，右半部分为右子树。先分别递归建立左子树和右子树，然后令根节点的指针分别指向两棵子树
//任意节点的左右子树的所有高度的差不大于1
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return build(0,nums.size()-1,nums);
    }
    TreeNode* build(int left,int right,vector<int>& nums)
    {
        if(left>right) return NULL;
        int mid=(left+right)/2;
        TreeNode* node=new TreeNode(nums[mid]);
        node->left=build(left,mid-1,nums);
        node->right=build(mid+1,right,nums);
        return node;
    }
};
```



### 230、二叉搜索树中第K小的元素

```c++
//二叉搜索树的中序就是升序
class Solution {
public:
    int rank=0;
    int res;   //存放值
    void search(TreeNode* root, int k)
    {
        if(root==NULL)
            return;
        search(root->left,k);
        rank++;
        if(rank==k)
        {
            res=root->val;
            return;
        }
        search(root->right,k);
    }
    int kthSmallest(TreeNode* root, int k) {
        search(root,k);
        return res;
    }
};
```

### 538、把二叉搜索树转化成累加树

```

class Solution {
public:
    int sum=0;
    void add_sum(TreeNode *root)
    {
        if(root==NULL) return;
        //从大到小的
        add_sum(root->right);
        sum+=root->val;
        root->val=sum;
        add_sum(root->left);

    }
    TreeNode* convertBST(TreeNode* root) {
        add_sum(root);
        return root;
    }
};
```

### 572、另一棵树的子树（递归）

```c++
/*
t 与 ss 相同；
tt 是 ss 左子树的子树；
tt 是 ss 右子树的子树。
*/
class Solution {
public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        if(root==nullptr) return false;
        return issame(root,subRoot)||isSubtree(root->left,subRoot)||isSubtree(root->right,subRoot);
    }
    bool issame(TreeNode* s,TreeNode* t)
    {
        if(!s&&!t) return true;
        if(!s&&t ||s&&!t || s->val!=t->val)
            return false;
        return issame(s->left,t->left)&&issame(s->right,t->right);
    }
};
```



### 700、二叉搜索树的搜索

```
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if(root==NULL) return NULL;
        if(root->val==val) return root;
        if(root->val<val)
        {
            return searchBST(root->right, val);
        }
       else
            return searchBST(root->left, val); 
    }
};
```

### 701、二叉搜索树中的插入操作

```
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        
        if(root==NULL)
            return new TreeNode(val);
        if(root->val>val)
        {
            root->left=insertIntoBST(root->left,val);
        }
        else       
            root->right=insertIntoBST(root->right,val);
            
        return root;
    }
};
```

### 98、验证二叉搜索树

```c++
//根据大小值判断
class Solution {
public:
    TreeNode* pre;
    bool isValidBST(TreeNode* root) {
        if(root==nullptr) return true;
        if(!isValidBST(root->left))
            return false;
        if(pre)
        {
            if(pre->val>=root->val)
                return false;
        }
        pre=root;
        if(!isValidBST(root->right))
            return false;
        return true;
    }
};
//解法一：递归
class Solution {
public:
    bool isValidBST(TreeNode* root,TreeNode*min,TreeNode*max) {
        if(root==NULL) return true;
        if(min!=NULL&&root->val<=min->val) return false;
        if(max!=NULL&&root->val>=max->val) return false;
        return isValidBST(root->left,min,root)&&isValidBST(root->right,root,max);
    }
    bool isValidBST(TreeNode* root) {
       return isValidBST(root,NULL,NULL);
    }

};
//方法二：转化成数组，看数组是否是递增元素
class Solution {
public:
     vector<int> vec;
    void traversal(TreeNode* root) {
        if (root == NULL) return;
        traversal(root->left);
        vec.push_back(root->val); // 将二叉搜索树转换为有序数组
        traversal(root->right);
    }
    bool isValidBST(TreeNode* root) {
        vec.clear();
       traversal(root);
       for (int i = 1; i < vec.size(); i++) {
            // 注意要小于等于，搜索树里不能有相同元素
            if (vec[i] <= vec[i - 1]) return false;
        }
       return true;
    }

};
// 根据左右子树的取值范围 
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return dfs(root,INT_MIN,INT_MAX);
    }
    bool dfs(TreeNode* root,long long min_num,long long max_num)
    {
        if(!root) return true;
        if(root->val<min_num||root->val>max_num) return false;
        return dfs(root->left,min_num,root->val-1ll)&&dfs(root->right,root->val+1ll,max_num);
    }
};
```

### 450、删除二叉树中的节点

```c++

class Solution {
public:
    
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root==NULL) return NULL;
        if(root->val==key)
        {
            if(root->left==NULL) return root->right;
            if(root->right==NULL) return root->left;
            //存在左右子树的情况
            //把要删除的节点的左子树放到被删除节点的右子树的最小节点的左子树位置上
            //并返回删除节点右孩子为新的根节点
            TreeNode *node=root->right;
            while(node->left!=NULL)
            {
                node=node->left;
            }
            node->left=root->left;
            root=root->right;
            return root;
           
        }
        else if(root->val>key)
        {
            root->left=deleteNode(root->left,key);
        }
        else
        {
            root->right=deleteNode(root->right,key);
        }
        return root;
    }
};
```

### 530、二叉搜索树的最小绝对差

```c++
//思路：先中序遍历，然后从一个数组里找差的最小值
class Solution {
public:
    vector<int> vec;
    void travel(TreeNode *root)
    {
        if(root==NULL) return;

        travel(root->left);
        vec.push_back(root->val);
        travel(root->right);

    }
    int getMinimumDifference(TreeNode* root) {
        travel(root);
        int result = INT_MAX;
        for(int i=1;i<vec.size();i++)
        {
            result=min(result,vec[i]-vec[i-1]);
        }
        return result;
    }
};
//在遍历的时候就计算
class Solution {
public:
    int ans=INT_MAX;
    int flag=1;  //是否是第一个节点
    int pre;
    void travel(TreeNode* root)
    {
        if(!root) return;
        travel(root->left);
        if(!flag)  ans=min(ans,root->val-pre);
        pre=root->val;
        flag=0;
        travel(root->right);
    }
    int getMinimumDifference(TreeNode* root) {
        travel(root);
        return ans;
    }
};
```

### 501、二叉搜索树中的众数

```
//没有利用搜索二叉树的性质
class Solution {
public:
   bool static cmp (const pair<int, int>& a, const pair<int, int>& b) {
    return a.second > b.second;
}

    void travel(TreeNode *root,unordered_map<int,int>&map)
    {
        if(root==NULL) return;
        travel(root->left,map);
        map[root->val]++;
        travel(root->right,map);

    }
    vector<int> findMode(TreeNode* root) {
        unordered_map<int,int> map;
        vector<int> result;
        if (root == NULL) return result;
        travel(root,map);
        //把map转化成vector
        vector<pair<int, int>> vec(map.begin(), map.end());
        sort(vec.begin(),vec.end(),cmp);
        result.push_back(vec[0].first);
        for (int i = 1; i < vec.size(); i++) { 
            // 取最高的放到result数组中
            if (vec[i].second == vec[0].second) result.push_back(vec[i].first);
            else break;
        }
        return result;
    }
};

//利用了二叉树的性质
//思路：中序遍历，比较当前节点是否和前一个节点的值相等，统计个数
class Solution {
public:
    TreeNode *pre; 
    int cnt;
    int maxcnt;
    vector<int> re;
    void travel(TreeNode *root)
    {
        if(root==NULL) return;
        travel(root->left);
        //中
        if(pre==NULL)  //第一个节点
        {
            cnt=1;
        }
        else if(pre->val==root->val)
        {
            cnt++;
        }
        else  //与前一个节点不同
        {
            cnt=1;
        }
        if(cnt==maxcnt)
        {
            re.push_back(root->val);
        }
        if(cnt>maxcnt)
        {
            maxcnt=cnt;
            re.clear();
            re.push_back(root->val);
        }
        pre=root;
        travel(root->right);

    }
    vector<int> findMode(TreeNode* root) {
        cnt = 0; 
        maxcnt = 0;
        TreeNode* pre = NULL; // 记录前一个节点
        re.clear();
        travel(root);
        return re;
    }
};
```

### 面试题34、二叉树中和为某一值的路径（递归、回溯）

```

class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    void backtracking(TreeNode *root,int tar)
    {
        if(root==NULL)
            return;
        path.push_back(root->val);
        tar-=root->val;
        if(tar==0&&root->left==NULL&&root->right==NULL)
            res.push_back(path);
        backtracking(root->left,tar);
        backtracking(root->right,tar);
        path.pop_back();
    }
    vector<vector<int>> pathSum(TreeNode* root, int target) {
            backtracking(root,target);
            return res;
    }
};
```

### 剑指 54、二叉搜索树的第K大节点

```c++
//按照右根左的顺序遍历
class Solution {
public:
    void dfs(TreeNode* root, int &k,int &result)
    {
        if(root==NULL) return;
        dfs(root->right,k,result);
        if (!--k) result = root->val;
        dfs(root->left,k,result);
    }
    int kthLargest(TreeNode* root, int k) {
        int result=0;
        dfs(root,k,result);
        return result;
    }
};
```

### 958、二叉树的完全性检验

```c++
//用一个标志记录是否是空节点
class Solution {
public:
    bool isCompleteTree(TreeNode* root) {
        //层序遍历
        queue<TreeNode*> que;
        if(!root) return true;
        que.push(root);
        bool end=0;   //代表已经遇到了空节点end=1
        while(!que.empty())
        {
            int n=que.size();
            for(int i=0;i<n;i++)
            {
                auto p=que.front();
                que.pop();
                if(!p)
                {
                    end=1;
                    continue;
                }
                //如果是完全树的话不会再执行一下的代码
                if(end) return false;
                que.push(p->left);
                que.push(p->right);
            }
        }
        return true;
    }
};
```

