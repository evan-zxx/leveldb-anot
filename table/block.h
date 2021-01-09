// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_TABLE_BLOCK_H_
#define STORAGE_LEVELDB_TABLE_BLOCK_H_

#include <cstddef>
#include <cstdint>

#include "leveldb/iterator.h"

namespace leveldb {

struct BlockContents;
class Comparator;

// sstable 中的数据以 block 单位存储，有利于 IO 和解析的粒度。
// sstable 的数据由一个个的 block 组成。当持久化数据时，多份 KV 聚合成 block 一次写入;
// 当读取时， 也是以 block 单位做 IO。
// sstable 的索引信息中会保存符合 key-range 的 block 在文件中的 offset/size(BlockHandle)。

//  entry: 一份 key-value 数据作为 block 内的一个 entry。考虑节约空间，leveldb 对 key 的存储 进行前缀压缩，
//  每个 entry 中会记录 key 与前一个 key 前缀相同的字节(shared_bytes)以及自 己独有的字节(unshared_bytes)。
//  读取时，对 block 进行遍历，每个 key 根据前一个 key 以及 shared_bytes/unshared_bytes 可以构造出来。

// 2) restarts: 如果完全按照 1)中所述处理，对每个 key 的查找，就都要从 block 的头开始遍历， 所以进一步细化粒度，对 block 内的前缀压缩分区段进行。
// 若干个 (Option::block_restart_interval)key 做前缀压缩之后，就重新开始下一轮。
// 每一轮前缀压 缩的 block offset 保存在 restarts 中，num_of_restarts 记录着总共压缩的轮数。

// 3) trailer:每个 block 后面都会有 5 个字节的 trailer。
// 1 个字节的 type 表示 block 内的数据是 否进行了压缩(比如使用了 snappy 压缩)，4 个字节的 crc 记录 block 数据的校验码。
class Block {
 public:
  // Initialize the block with the specified contents.
  explicit Block(const BlockContents& contents);

  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  ~Block();

  size_t size() const { return size_; }
  Iterator* NewIterator(const Comparator* comparator);

 private:
  class Iter;

  uint32_t NumRestarts() const;

  const char* data_;
  size_t size_;
  uint32_t restart_offset_;  // Offset in data_ of restart array
  bool owned_;               // Block owns data_[]
};

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_TABLE_BLOCK_H_
