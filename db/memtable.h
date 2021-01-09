// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_DB_MEMTABLE_H_
#define STORAGE_LEVELDB_DB_MEMTABLE_H_

#include <string>

#include "db/dbformat.h"
#include "db/skiplist.h"
#include "leveldb/db.h"
#include "util/arena.h"

namespace leveldb {

class InternalKeyComparator;
class MemTableIterator;

// 类似 BigTable 的模式，数据在内存中以 memtable 形式存储。
// leveldb 的 memtable 实现没有使用复 杂的 B-树系列，采用的是更轻量级的 skip list。
// 全局看来，skip list 所有的 node 就是一个排序的链表，考虑到操作效率，为这一个链表再添加若 干不同跨度的辅助链表，查找时通过辅助链表可以跳跃比较来加大查找的步进。
// 每个链表上都是排序的node，而每个node也可能同时处在多个链表上。将一个 node 所属链表的数量看作它的高度，那 么，不同高度的 node 在查找时会获得不同跳跃跨度的查找优化，图 1 是一个最大高度为 5 的 skiplist。
// 换个角度，如果 node 的高度具有随机性，数据集合从高度层次上看就有了散列性，也就等同于树的 平衡。
// 相对于其他树型数据结构采用不同策略来保证平衡状态，Skip list 仅保证新加入 node 的高 度随机即可(当然也可以采用规划计算的方式确定高度，以获得平摊复杂度。leveldb 采用的是更简单的方式)
// 如前所述，作为随机性的数据机构，skip list 的算法复杂度依赖于我们的随机假设，复杂度为 O (logn).
// 基于下面两个特点，skiplist 中的操作不需要任何锁或者 node 的引用计数:
// 1) skip list 中 node 内保存的是 InternalKey 与相应 value 组成的数据，SequnceNumber 的全局唯一保证了不会有相同的node出现，也就保证了不会有node更新的情况。
// 2) delete 等同于 put 操作，所以不会需要引用计数记录 node 的存活周期。
class MemTable {
 public:
  // MemTables are reference counted.  The initial reference count
  // is zero and the caller must call Ref() at least once.
  explicit MemTable(const InternalKeyComparator& comparator);

  MemTable(const MemTable&) = delete;
  MemTable& operator=(const MemTable&) = delete;

  // Increase reference count.
  void Ref() { ++refs_; }

  // Drop reference count.  Delete if no more references exist.
  void Unref() {
    --refs_;
    assert(refs_ >= 0);
    if (refs_ <= 0) {
      delete this;
    }
  }

  // Returns an estimate of the number of bytes of data in use by this
  // data structure. It is safe to call when MemTable is being modified.
  size_t ApproximateMemoryUsage();

  // Return an iterator that yields the contents of the memtable.
  //
  // The caller must ensure that the underlying MemTable remains live
  // while the returned iterator is live.  The keys returned by this
  // iterator are internal keys encoded by AppendInternalKey in the
  // db/format.{h,cc} module.
  Iterator* NewIterator();

  // Add an entry into memtable that maps key to value at the
  // specified sequence number and with the specified type.
  // Typically value will be empty if type==kTypeDeletion.
  void Add(SequenceNumber seq, ValueType type, const Slice& key,
           const Slice& value);

  // If memtable contains a value for key, store it in *value and return true.
  // If memtable contains a deletion for key, store a NotFound() error
  // in *status and return true.
  // Else, return false.
  bool Get(const LookupKey& key, std::string* value, Status* s);

 private:
  friend class MemTableIterator;
  friend class MemTableBackwardIterator;

  struct KeyComparator {
    const InternalKeyComparator comparator;
    explicit KeyComparator(const InternalKeyComparator& c) : comparator(c) {}
    int operator()(const char* a, const char* b) const;
  };

  typedef SkipList<const char*, KeyComparator> Table;

  ~MemTable();  // Private since only Unref() should be used to delete it

  KeyComparator comparator_;
  int refs_;
  Arena arena_;
  Table table_;
};

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_DB_MEMTABLE_H_
