//------------------------------------------------------------------------------
//
//   Copyright 2018-2019 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "vm_test_suite.hpp"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>

class CustomBindingTests : public VmTestSuite
{
protected:
};

// Test to add a custom binding that will increment this counter when
// the smart contract is executed
static int32_t binding_called_count = 0;

static void CustomBinding(fetch::vm::VM * /*vm*/)
{
  binding_called_count++;
}

TEST_F(CustomBindingTests, CheckBasicBinding)
{
  static char const *TEXT =
      " function main() "
      "   customBinding();"
      " endfunction ";

  EXPECT_EQ(binding_called_count, 0);

  // create the binding
  module_->CreateFreeFunction("customBinding", &CustomBinding);

  ASSERT_TRUE(Compile(TEXT));

  ASSERT_TRUE(Run());
  ASSERT_TRUE(Run());
  ASSERT_TRUE(Run());

  EXPECT_EQ(binding_called_count, 3);
}